# coding=utf-8
"""Samplers for the latent diffusion model.

The samplers in this module generate latent codes from pure noise with a
trained noise-prediction network. They follow the calling convention of
`train.sample_latents` so that they can be swapped through the config, and they
receive the network as a `predict_fn` argument so that this module does not
depend on `train.py` (which avoids a circular import).

Notation used throughout.
  T: number of training timesteps (`config.timestep`, 1000 by default).
  t: integer timestep in {1, ..., T}. The noise level at timestep t is
    `alpha_bar[t - 1]` (see `diffusion.get_diffusion_parameters`).
  alpha_bar_t: cumulative product of the alphas up to timestep t. The noisy
    latent is z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps, where
    eps is Gaussian noise with standard deviation `sigma` in this codebase.
"""
from typing import Any, Callable, Tuple

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from mesh_diffusion.sin_im_ddm import diffusion as df


def make_timestep_sequence(
    timestep: int,
    num_sampling_steps: int,
    spacing: str = 'trailing',
) -> np.ndarray:
  """Builds the decreasing list of diffusion timesteps visited by a sampler.

  The network was trained on the integer timesteps {1, ..., T}. A sampler that
  performs N < T steps only visits N of them. Two spacings are supported.

  'leading' follows the original DDIM implementation (and `df.get_alphas`). The
  timesteps are 1, 1 + k, 1 + 2k, ... with k = T // N, visited in decreasing
  order. The first timestep is 1 + (N - 1) * k, which is smaller than T.

  'trailing' makes the first timestep equal to T, so that sampling really starts
  from the noise level the network was trained to handle as pure noise. The
  timesteps are round(i * T / N) for i = N, N - 1, ..., 1.

  With N = T both spacings return T, T - 1, ..., 1, which are the timesteps of
  ancestral (DDPM) sampling.

  Args:
    timestep: Number of training timesteps T.
    num_sampling_steps: Number of sampler steps N, with 1 <= N <= T.
    spacing: Either 'leading' or 'trailing'.

  Returns:
    An integer array of shape (N,), strictly decreasing, with every entry in
    [1, T].

  Raises:
    ValueError: If N is not in [1, T] or if the spacing is unknown.
  """

  if num_sampling_steps > timestep or num_sampling_steps < 1:
    raise ValueError("N should be in [1,T]")

  if spacing not in ('leading', 'trailing'):
    raise ValueError('spacing sould be either leading or trailing')

  if spacing == 'leading':
    k = timestep // num_sampling_steps
    i = np.arange(num_sampling_steps - 1, -1, -1)
    return 1 + k * i

  else:
    i = np.arange(num_sampling_steps, 0, -1)
    return np.round(i * timestep / num_sampling_steps).astype(int)


def noise_levels_for_sequence(
    alpha_bar: Array,
    ts: np.ndarray,
) -> Tuple[Array, Array]:
  """Gets the noise level of every visited timestep and of its successor.

  A sampler step goes from timestep ts[i] to the next visited timestep
  ts[i + 1]. The last step goes to timestep 0, which is the clean latent and
  has noise level alpha_bar = 1.

  Args:
    alpha_bar: Array of shape (T,) with the noise levels of the timesteps
      1, ..., T, so that the level of timestep t is alpha_bar[t - 1].
    ts: Integer array of shape (N,) with the visited timesteps, in decreasing
      order, as returned by `make_timestep_sequence`.

  Returns:
    A tuple (alpha_bar_t, alpha_bar_prev). Both have shape (N,).
    alpha_bar_t[i] is the noise level of ts[i]. alpha_bar_prev[i] is the noise
    level of the timestep reached by step i, equal to 1.0 for the last step.
  """

  alpha_bar_t = alpha_bar[ts - 1]
  alpha_bar_prev = jnp.concatenate((alpha_bar_t[1:], np.asarray([1.0])), axis=0)
  return (alpha_bar_t, alpha_bar_prev)

def ddim_step(
    alpha_bar_t: Array,
    alpha_bar_prev: Array,
    zt: Array,
    epst: Array,
    noise: Array,
    eta: float = 0.0,
) -> Array:
  """Performs one DDIM update from noise level alpha_bar_t to alpha_bar_prev.

  The update first estimates the clean latent from the predicted noise, then
  recombines this estimate with the predicted noise at the lower noise level.
  The coefficient `eta` interpolates between the deterministic DDIM sampler
  (eta = 0) and ancestral DDPM sampling (eta = 1, when the two noise levels
  belong to consecutive timesteps). See Song et al., "Denoising Diffusion
  Implicit Models", 2021, Eq. 12.

  The function is elementwise. It works for any array shape and for real or
  complex arrays, as long as all array arguments share the same shape.

  Args:
    alpha_bar_t: Noise level of `zt` (scalar or broadcastable to `zt`).
    alpha_bar_prev: Noise level of the output latent, greater than
      `alpha_bar_t`. A value of 1.0 means that the output is a clean latent.
    zt: Noisy latent at noise level `alpha_bar_t`.
    epst: Noise predicted by the network for `zt`. It has the same scale as the
      noise that was added during training, that is `sigma` times a standard
      Gaussian.
    noise: Fresh noise used when eta > 0. It must already be multiplied by
      `sigma`. It is ignored when eta = 0.
    eta: Stochasticity of the update, in [0, 1].

  Returns:
    The latent at noise level `alpha_bar_prev`, with the same shape as `zt`.
  """

  # 1. Clean latent estimated from z_t = sqrt(ab_t) z0 + sqrt(1 - ab_t) eps.
  z0_hat = (zt - jnp.sqrt(1.0 - alpha_bar_t) * epst) / jnp.sqrt(alpha_bar_t)

  # 2. Standard deviation of the injected noise (Song et al., Eq. 16).
  sigma_t = (
      eta
      * jnp.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
      * jnp.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
  )

  # 3. Recombine at noise level alpha_bar_prev. The clamp protects the square
  # root against tiny negative values caused by rounding.
  dir_coef = jnp.sqrt(jnp.maximum(1.0 - alpha_bar_prev - sigma_t**2, 0.0))
  return jnp.sqrt(alpha_bar_prev) * z0_hat + dir_coef * epst + sigma_t * noise


def sample_latents_ddim(
    predict_fn: Callable[..., Array],
    geom_data: Any,
    params: Any,
    key: Array,
    shape: Tuple[int, ...],
    timestep: int,
    sigma: float = 1.0,
    schedule: str = 'cos',
    num_sampling_steps: int = 50,
    eta: float = 0.0,
    spacing: str = 'trailing',
    inpaint_data: Any = None,
) -> Tuple[Array, Array]:
  """Generates latent codes with the DDIM sampler.

  The function mirrors `train.sample_latents` (same inputs, same outputs) but
  visits only `num_sampling_steps` timesteps instead of all of them.

  Args:
    predict_fn: Function with signature
      `predict_fn(params=..., geom_data=..., zt=..., t=...)` that returns the
      noise predicted by the network. It is meant to be the pmapped version of
      `train.predict`, built in the caller with
      `jax.pmap(functools.partial(predict, model=model), axis_name='batch')`.
      `zt` is a complex array and `t` is a float array with one entry per
      device. The output is a complex array with the shape of `zt`.
    geom_data: Geometry inputs of the network, replicated over devices.
    params: Network parameters, replicated over devices.
    key: JAX random key.
    shape: Shape of the latent codes in the real representation, that is
      (num_devices, num_verts, latent_dim, 2) where the last axis holds the real
      and imaginary parts.
    timestep: Number of training timesteps T.
    sigma: Noise scale used during training (see `train.train_and_evaluate`).
    schedule: Name of the noise schedule, as in `df.get_diffusion_parameters`.
    num_sampling_steps: Number of network evaluations N.
    eta: DDIM stochasticity in [0, 1]. Zero gives a deterministic sampler.
    spacing: Timestep spacing, 'leading' or 'trailing'.
    inpaint_data: Reserved for inpainting. Must be None for now.

  Returns:
    A tuple (latents, key). `latents` is a complex array of shape
    (num_devices, num_verts, latent_dim) and `key` is the updated random key.

  Raises:
    NotImplementedError: If `inpaint_data` is not None.
  """
  if inpaint_data is not None:
    raise NotImplementedError('Inpainting is not supported by DDIM yet.')

  _, _, alpha_bar, _, _ = df.get_diffusion_parameters(timestep, schedule)

  ts = make_timestep_sequence(timestep, num_sampling_steps, spacing)
  alpha_bar_t, alpha_bar_prev = noise_levels_for_sequence(alpha_bar, ts)

  key, init_key = jax.random.split(key)
  zt = sigma * jax.random.normal(init_key, shape)

  for i, t in enumerate(ts):
    # Time input of the network: one float per device.
    time_vec = float(t) * jnp.ones(zt.shape[0], dtype=jnp.float32)

    # The network works on complex latents, the sampler on (real, imag) pairs.
    zt_c = jax.lax.complex(zt[..., 0], zt[..., 1])
    epst = predict_fn(params=params, geom_data=geom_data, zt=zt_c, t=time_vec)
    epst = jnp.stack((jnp.real(epst), jnp.imag(epst)), axis=-1)

    # The deterministic sampler (eta = 0) does not consume random numbers.
    if eta > 0.0:
      key, noise_key = jax.random.split(key)
      noise = sigma * jax.random.normal(noise_key, zt.shape)
    else:
      noise = jnp.zeros_like(zt)

    zt = ddim_step(alpha_bar_t[i], alpha_bar_prev[i], zt, epst, noise, eta)

  return jax.lax.complex(zt[..., 0], zt[..., 1]), key
