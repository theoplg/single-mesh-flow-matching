# coding=utf-8
"""Flow matching for the latent texture model.

This module replaces the diffusion objective by flow matching while keeping the
network, the latent codes and the calling conventions of `train.py` and of
`mesh_diffusion/ddim/samplers.py`.

Notation used throughout.
  z1: clean latent code (the data), complex array.
  z0: Gaussian noise with standard deviation `sigma` on the real and on the
    imaginary part, so that its scale matches the scale of the latent codes.
  tau: time in [0, 1]. tau = 0 is pure noise and tau = 1 is the clean latent.
  x_tau = (1 - tau) * z0 + tau * z1: point at time tau on the straight path.
  v = z1 - z0: velocity of that path. The network is trained to predict it.
  time_scale: factor applied to the flipped time 1 - tau before it is given to the
    network (see `network_time`), so that
    the network receives values in [0, time_scale). The default 1000 keeps the
    range of the integer timesteps of the diffusion model (1, ..., 1000), which
    the sinusoidal time embedding of the network was designed for.

Isometry equivariance. A rotation of the tangent frame at a vertex multiplies the
complex latents of this vertex by a unit complex number. Both z0 (circular
complex Gaussian noise) and z1 are transformed in the same way, so x_tau and v
are transformed in the same way too. The equivariance of the network therefore
applies to the flow matching objective exactly as it does to the diffusion one.
"""
from typing import Any, Callable, Tuple

import jax
from jax import Array
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import optax


def _expand_like(tau: Array, x: Array) -> Array:
  """Reshapes `tau` so that it broadcasts against `x` along the leading axes."""
  tau = jnp.asarray(tau)
  return jnp.reshape(tau, tau.shape + (1,) * (x.ndim - tau.ndim))


def interpolate(z0: Array, z1: Array, tau: Array) -> Array:
  """Gets the point at time tau on the straight path from z0 to z1.

  The function is elementwise. It works for real or complex arrays.

  Args:
    z0: Noise (the point at tau = 0), array of shape (B, ...) or any shape.
    z1: Data (the point at tau = 1), same shape as `z0`.
    tau: Time, either a scalar or an array of shape (B,)

  Returns:
    The point x_tau = (1 - tau) * z0 + tau * z1, with the shape of `z0`.
  """
  tau = _expand_like(tau, z0)
  return (1-tau)*z0  + tau*z1  


def network_time(tau: Array, time_scale: float = 1000.0) -> Array:
  """Gets the time given to the network at flow time tau.

  The diffusion network was trained with large times for noisy latents (t = 1000 is
  pure noise). The flow time runs the other way (tau = 0 is pure noise), so tau is
  flipped. The time input then keeps the same meaning as in the diffusion model,
  which lets a warm start reuse the pretrained time embedding. Training and
  sampling must both use this function.

  Args:
    tau: Flow time in [0, 1], scalar or array.
    time_scale: Value given to the network for pure noise (tau = 0).

  Returns:
    `time_scale * (1 - tau)`, which is `time_scale` for noise and 0 for data.
  """
  return time_scale * (1.0 - tau)


def make_flow_training_batch(
    z: Array,
    key: Array,
    sigma: float = 1.0,
    time_scale: float = 1000.0,
) -> Tuple[Array, Array, Array, Array]:
  """Builds a flow matching training batch from clean latent codes.

  It plays the role of `diffusion.encode_batched_tangent_latent_samples` and has
  the same inputs and outputs, so that the training loop of `train.py` only needs
  to switch between the two functions.

  Args:
    z: Clean latent codes, complex array of shape (B, num_verts, latent_dim). B is
      the number of devices, each device receives one mesh.
    key: JAX random key.
    sigma: Scale of the noise, as in `train.train_and_evaluate`.
    time_scale: Factor applied to tau to get the time given to the network.

  Returns:
    A tuple (xt, target, t, key).
      xt: Latents at time tau, complex, with the shape of `z`.
      target: Velocity to be predicted by the network, complex, with the shape of
        `z`.
      t: Time given to the network, float array of shape (B,) equal to
        `time_scale * (1 - tau)`, see `network_time`.
      key: Updated random key.
  """
  batch_size = z.shape[0]
  key, tau_key, noise_key = jax.random.split(key, num=3)

  tau = jax.random.uniform(tau_key, (batch_size,), minval=0.0, maxval=1.0)

  eps = sigma * jax.random.normal(noise_key, shape=z.shape + (2,))
  z0 = jax.lax.complex(eps[..., 0], eps[..., 1])

  xt = interpolate(z0, z, _expand_like(tau, z))
  target = z - z0
  t = network_time(tau, time_scale)

  return xt, target, t, key


def flow_loss(pred: Array, target: Array, loss_type: str = 'l1') -> Array:
  """Mean error between the predicted and the target velocity.

  Args:
    pred: Predicted velocity, complex.
    target: Target velocity, complex, same shape as `pred`.
    loss_type: 'l1' averages the modulus of the error, as the diffusion loss of
      `train.train_step`. 'l2' averages the squared modulus.

  Returns:
    A scalar.

  Raises:
    ValueError: If `loss_type` is unknown.
  """
  err = jnp.abs(pred - target)  # modulus of the complex error.
  if loss_type == 'l1':
    return jnp.mean(err)
  if loss_type == 'l2':
    return jnp.mean(err**2)
  raise ValueError(f'Unknown loss_type {loss_type!r}, expected l1 or l2.')


def euler_step(x: Array, v: Array, dt: float) -> Array:
  """Explicit Euler step of the flow ODE dx/dtau = v(x, tau).

  Args:
    x: Current state.
    v: Velocity at the current state and time.
    dt: Time increment (positive).

  Returns:
    The state after the step.
  """
  return x + dt * v


def heun_step(x: Array, v_start: Array, v_end: Array, dt: float) -> Array:
  """Heun (trapezoidal) step of the flow ODE.

  Args:
    x: Current state.
    v_start: Velocity at the current state and time.
    v_end: Velocity at the Euler prediction of the next state, at the next time.
    dt: Time increment (positive).

  Returns:
    The state after the step, obtained with the average of the two velocities.
  """
  return x + 0.5 * dt * (v_start + v_end)


def make_time_grid(num_steps: int) -> np.ndarray:
  """Returns the num_steps + 1 times 0 = tau_0 < ... < tau_N = 1 of the sampler."""
  if num_steps < 1:
    raise ValueError('num_steps should be at least 1')
  return np.linspace(0.0, 1.0, num_steps + 1)


def sample_latents_flow(
    predict_fn: Callable[..., Array],
    geom_data: Any,
    params: Any,
    key: Array,
    shape: Tuple[int, ...],
    sigma: float = 1.0,
    num_sampling_steps: int = 50,
    solver: str = 'euler',
    time_scale: float = 1000.0,
    inpaint_data: Any = None,
) -> Tuple[Array, Array]:
  """Generates latent codes by integrating the learned flow from noise to data.

  The function mirrors `samplers.sample_latents_ddim` (same inputs, same outputs)
  so that it can be selected through the config.

  Args:
    predict_fn: Function with signature
      `predict_fn(params=..., geom_data=..., zt=..., t=...)` that returns the
      velocity predicted by the network. It is the same pmapped `train.predict` as
      for the DDIM sampler. `zt` is a complex array and `t` is a float array with
      one entry per device.
    geom_data: Geometry inputs of the network, replicated over devices.
    params: Network parameters, replicated over devices.
    key: JAX random key.
    shape: Shape of the latent codes in the real representation, that is
      (num_devices, num_verts, latent_dim, 2) where the last axis holds the real
      and imaginary parts.
    sigma: Noise scale used during training.
    num_sampling_steps: Number of intervals N of the time grid.
    solver: 'euler' (N network evaluations) or 'heun' (2 N - 1 evaluations).
    time_scale: Factor applied to tau to get the time given to the network. It must
      be the value used during training.
    inpaint_data: Reserved for inpainting. Must be None for now.

  Returns:
    A tuple (latents, key). `latents` is a complex array of shape
    (num_devices, num_verts, latent_dim) and `key` is the updated random key.

  Raises:
    NotImplementedError: If `inpaint_data` is not None.
    ValueError: If the solver is unknown.
  """
  if inpaint_data is not None:
    raise NotImplementedError('Inpainting is not supported by flow matching yet.')
  if solver not in ('euler', 'heun'):
    raise ValueError('solver should be either euler or heun')

  grid = make_time_grid(num_sampling_steps)
  key, init_key = jax.random.split(key)
  x = sigma * jax.random.normal(init_key, shape)

  def velocity(x, tau):
    """Velocity predicted by the network at time tau, on (real, imag) pairs."""
    time_vec = network_time(float(tau), time_scale) * jnp.ones(
        x.shape[0], dtype=jnp.float32
    )
    x_c = jax.lax.complex(x[..., 0], x[..., 1])
    v = predict_fn(params=params, geom_data=geom_data, zt=x_c, t=time_vec)
    return jnp.stack((jnp.real(v), jnp.imag(v)), axis=-1)

  num_intervals = len(grid) - 1
  for i in range(num_intervals):
    tau, tau_next = grid[i], grid[i + 1]
    dt = tau_next - tau
    v_start = velocity(x, tau)
    x_euler = euler_step(x, v_start, dt)
    if solver == 'heun' and i < num_intervals - 1:
      v_end = velocity(x_euler, tau_next)
      x = heun_step(x, v_start, v_end, dt)
    else:
      x = x_euler

  return jax.lax.complex(x[..., 0], x[..., 1]), key


def train_step(
    model: Any,
    optimizer: optax.GradientTransformation,
    geom_data: Any,
    state: Any,
    zt: Array,
    target: Array,
    t: Any,
    metrics_cls: Any,
    loss_type: str = 'l1',
) -> Tuple[Any, Any]:
  """Performs a single flow matching training step (to be pmapped).

  Same as `train.train_step` except for the loss, which is `flow_loss`.

  Args:
    model: Flax module of the network.
    optimizer: Optax optimizer.
    geom_data: Geometry inputs of the network.
    state: Train state (needs `params`, `opt_state`, `step` and `replace`).
    zt: Latents at time tau, complex.
    target: Target velocity, complex.
    t: Time given to the network (`time_scale * (1 - tau)`).
    metrics_cls: Metrics collection of `train.py`, which provides
      `single_from_model_output`.
    loss_type: 'l1' or 'l2'.

  Returns:
    The new state and the metrics of the step.
  """

  def get_loss(params):
    out = model.apply({'params': params}, zt, t, *geom_data)
    if isinstance(out, tuple):
      out = out[0]
    return flow_loss(out, target, loss_type)

  params = state.params
  loss_val, grads = jax.value_and_grad(get_loss)(params)

  grads = jax.lax.pmean(grads, axis_name='batch')
  grads = jax.tree_util.tree_map(jnp.conj, grads)

  updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)

  gravel = jnp.abs(ravel_pytree(grads)[0])
  uravel = jnp.abs(ravel_pytree(updates)[0])

  new_state = state.replace(
      step=state.step + 1, params=new_params, opt_state=new_opt_state
  )
  metrics_update = metrics_cls.single_from_model_output(
      loss=loss_val,
      mean_grads=jnp.mean(gravel),
      max_grads=jnp.max(gravel),
      max_updates=jnp.max(uravel),
      mean_updates=jnp.mean(uravel),
  )
  return new_state, metrics_update