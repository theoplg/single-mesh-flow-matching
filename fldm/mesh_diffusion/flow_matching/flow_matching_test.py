# coding=utf-8
"""Unit tests for `flow_matching`. They run on CPU in a few seconds.

Run from the `fldm/` directory with
  JAX_PLATFORMS=cpu python -m mesh_diffusion.flow_matching.flow_matching_test
"""
import functools

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mesh_diffusion.flow_matching import flow_matching as fm

SHAPE_C = (2, 50, 8)  # (batch, vertices, latent_dim), complex.
TIME_SCALE = 1000.0


def _flow_time(t):
  """Inverse of `fm.network_time`: flow time tau from the network time t."""
  return 1.0 - t / TIME_SCALE


def _complex_normal(key, shape):
  noise = jax.random.normal(key, shape + (2,))
  return jax.lax.complex(noise[..., 0], noise[..., 1])


class InterpolateTest(absltest.TestCase):

  def test_endpoints_and_midpoint(self):
    z0 = _complex_normal(jax.random.PRNGKey(0), SHAPE_C)
    z1 = _complex_normal(jax.random.PRNGKey(1), SHAPE_C)
    ones = jnp.ones(2)
    np.testing.assert_allclose(fm.interpolate(z0, z1, 0 * ones), z0, atol=1e-6)
    np.testing.assert_allclose(fm.interpolate(z0, z1, ones), z1, atol=1e-6)
    np.testing.assert_allclose(
        fm.interpolate(z0, z1, 0.5 * ones), 0.5 * (z0 + z1), atol=1e-6)

  def test_one_time_per_batch_element(self):
    z0 = jnp.zeros(SHAPE_C, dtype=jnp.complex64)
    z1 = jnp.ones(SHAPE_C, dtype=jnp.complex64)
    out = fm.interpolate(z0, z1, jnp.array([0.25, 0.75]))
    np.testing.assert_allclose(out[0], 0.25, atol=1e-6)
    np.testing.assert_allclose(out[1], 0.75, atol=1e-6)


class TrainingBatchTest(absltest.TestCase):

  def test_shapes_dtypes_and_time_range(self):
    z = _complex_normal(jax.random.PRNGKey(0), (64, 10, 8))
    xt, target, t, key = fm.make_flow_training_batch(
        z, jax.random.PRNGKey(1), sigma=2.0, time_scale=1000.0)
    self.assertEqual(xt.shape, z.shape)
    self.assertEqual(target.shape, z.shape)
    self.assertTrue(jnp.iscomplexobj(xt) and jnp.iscomplexobj(target))
    self.assertEqual(t.shape, (64,))
    # tau in [0, 1) gives t = 1000 * (1 - tau) in (0, 1000].
    self.assertTrue(bool(jnp.all((t > 0) & (t <= 1000))))
    self.assertEqual(key.shape, jax.random.PRNGKey(1).shape)

  def test_data_is_recovered_from_xt_and_target(self):
    # x_tau + (1 - tau) * v = z1 for the straight path.
    z = _complex_normal(jax.random.PRNGKey(0), SHAPE_C)
    xt, target, t, _ = fm.make_flow_training_batch(z, jax.random.PRNGKey(1))
    tau = _flow_time(t)[:, None, None]
    np.testing.assert_allclose(xt + (1.0 - tau) * target, z, atol=1e-5)

  def test_noise_has_scale_sigma_and_is_circular(self):
    sigma = 3.0
    z = jnp.zeros((200, 100, 8), dtype=jnp.complex64)
    xt, target, t, _ = fm.make_flow_training_batch(
        z, jax.random.PRNGKey(0), sigma=sigma)
    tau = _flow_time(t)[:, None, None]
    noise = xt - tau * target  # equals z0 when z = 0, up to rounding.
    np.testing.assert_allclose(jnp.std(noise.real), sigma, rtol=0.02)
    np.testing.assert_allclose(jnp.std(noise.imag), sigma, rtol=0.02)
    np.testing.assert_allclose(target, -noise.reshape(target.shape), atol=1e-4)

  def test_times_are_uniform(self):
    z = jnp.zeros((20000, 1, 1), dtype=jnp.complex64)
    _, _, t, _ = fm.make_flow_training_batch(z, jax.random.PRNGKey(0))
    tau = _flow_time(t)
    np.testing.assert_allclose(jnp.mean(tau), 0.5, atol=0.01)
    np.testing.assert_allclose(jnp.var(tau), 1.0 / 12.0, atol=0.005)

  def test_deterministic_given_key_and_new_key_returned(self):
    z = _complex_normal(jax.random.PRNGKey(0), SHAPE_C)
    a = fm.make_flow_training_batch(z, jax.random.PRNGKey(5))
    b = fm.make_flow_training_batch(z, jax.random.PRNGKey(5))
    for x, y in zip(a, b):
      np.testing.assert_array_equal(x, y)
    self.assertFalse(np.array_equal(a[3], jax.random.PRNGKey(5)))


class NetworkTimeTest(absltest.TestCase):

  def test_flipped_time(self):
    # Pure noise (tau = 0) gets the largest time, as in the diffusion model.
    np.testing.assert_allclose(
        fm.network_time(jnp.array([0.0, 0.25, 1.0])), [1000.0, 750.0, 0.0])
    np.testing.assert_allclose(fm.network_time(0.5, time_scale=10.0), 5.0)


class LossTest(absltest.TestCase):

  def test_l1_and_l2_on_complex_values(self):
    target = jnp.zeros((4, 3), dtype=jnp.complex64)
    pred = jnp.full((4, 3), 3.0 + 4.0j, dtype=jnp.complex64)  # modulus 5.
    np.testing.assert_allclose(fm.flow_loss(pred, target, 'l1'), 5.0, rtol=1e-6)
    np.testing.assert_allclose(fm.flow_loss(pred, target, 'l2'), 25.0, rtol=1e-6)

  def test_zero_when_prediction_is_exact(self):
    x = _complex_normal(jax.random.PRNGKey(0), SHAPE_C)
    self.assertEqual(float(fm.flow_loss(x, x, 'l1')), 0.0)

  def test_unknown_loss_raises(self):
    x = jnp.zeros(3)
    with self.assertRaises(ValueError):
      fm.flow_loss(x, x, 'huber')


class StepTest(absltest.TestCase):

  def test_euler_and_heun(self):
    x, v1, v2 = jnp.array(1.0), jnp.array(2.0), jnp.array(4.0)
    np.testing.assert_allclose(fm.euler_step(x, v1, 0.5), 2.0)
    np.testing.assert_allclose(fm.heun_step(x, v1, v2, 0.5), 1.0 + 0.5 * 3.0)

  def test_time_grid(self):
    grid = fm.make_time_grid(4)
    np.testing.assert_allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    with self.assertRaises(ValueError):
      fm.make_time_grid(0)


class SamplerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.shape = (1, 30, 8, 2)
    self.target = _complex_normal(jax.random.PRNGKey(7), self.shape[:-1])
    self.calls = []

    def oracle(params, geom_data, zt, t):
      # Exact velocity of the straight path when the data is a single point z1:
      # v(x, tau) = (z1 - x) / (1 - tau).
      del params, geom_data
      tau = _flow_time(t)[:, None, None]
      self.calls.append(float(t[0]))
      return (self.target - zt) / (1.0 - tau)

    self.oracle = oracle

  def _sample(self, n, solver, seed=0, sigma=1.0):
    return fm.sample_latents_flow(
        predict_fn=self.oracle, geom_data=None, params=None,
        key=jax.random.PRNGKey(seed), shape=self.shape, sigma=sigma,
        num_sampling_steps=n, solver=solver)

  def test_euler_reaches_the_data_with_the_exact_velocity(self):
    for n in (1, 5, 40):
      out, _ = self._sample(n, 'euler')
      self.assertEqual(out.shape, self.shape[:-1])
      self.assertTrue(jnp.iscomplexobj(out))
      np.testing.assert_allclose(out, self.target, atol=1e-4)

  def test_heun_reaches_the_data_with_the_exact_velocity(self):
    for n in (1, 5, 40):
      out, _ = self._sample(n, 'heun')
      np.testing.assert_allclose(out, self.target, atol=1e-4)

  def test_number_of_network_evaluations(self):
    self._sample(10, 'euler')
    self.assertLen(self.calls, 10)
    self.calls.clear()
    self._sample(10, 'heun')
    self.assertLen(self.calls, 19)  # 2 N - 1.

  def test_times_given_to_the_network(self):
    self._sample(4, 'euler')
    # tau = 0, 0.25, 0.5, 0.75, flipped by `fm.network_time`.
    np.testing.assert_allclose(self.calls, [1000.0, 750.0, 500.0, 250.0])

  def test_deterministic_and_key_dependent_start(self):
    # A network that does not move the state exposes the initial noise.
    def still(params, geom_data, zt, t):
      del params, geom_data, t
      return jnp.zeros_like(zt)

    def run(seed, sigma):
      out, _ = fm.sample_latents_flow(
          still, None, None, jax.random.PRNGKey(seed), self.shape,
          sigma=sigma, num_sampling_steps=3)
      return out

    np.testing.assert_array_equal(run(0, 1.0), run(0, 1.0))
    self.assertFalse(np.array_equal(run(0, 1.0), run(1, 1.0)))
    np.testing.assert_allclose(jnp.std(run(0, 2.0).real) / jnp.std(run(0, 1.0).real),
                               2.0, rtol=1e-5)

  def test_invalid_arguments(self):
    with self.assertRaises(NotImplementedError):
      fm.sample_latents_flow(
          self.oracle, None, None, jax.random.PRNGKey(0), self.shape,
          inpaint_data=(None, None))
    with self.assertRaises(ValueError):
      self._sample(5, 'rk4')


@flax.struct.dataclass
class _State:
  step: int
  params: dict
  opt_state: optax.OptState


class _Metrics:

  @staticmethod
  def single_from_model_output(**kwargs):
    return kwargs


class _Scale:
  """Toy model: out = w * zt, where the time input and the geometry are unused."""

  def apply(self, variables, zt, t, *geom_data):
    del t, geom_data
    return variables['params']['w'] * zt


class TrainStepTest(absltest.TestCase):
  """Uses a complex parameter, like the network, to check the gradient convention."""

  def _run(self, loss_type):
    optimizer = optax.sgd(0.1)
    params = {'w': jnp.array(0.0 + 0.0j, dtype=jnp.complex64)}
    state = _State(0, params, optimizer.init(params))
    zt = jnp.ones((1, 5), dtype=jnp.complex64)
    target = (2.0 + 2.0j) * jnp.ones((1, 5), dtype=jnp.complex64)
    t = jnp.zeros((1,))
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)
    step = jax.pmap(
        functools.partial(
            fm.train_step, model=_Scale(), optimizer=optimizer, geom_data=(),
            metrics_cls=_Metrics, loss_type=loss_type),
        axis_name='batch')
    return step(state=state, zt=zt, target=target, t=t)

  def test_l2_loss_and_update_direction(self):
    new_state, metrics = self._run('l2')
    np.testing.assert_allclose(metrics['loss'], 8.0, rtol=1e-5)  # |0 - (2+2j)|^2.
    # The loss is (a - 2)^2 + (b - 2)^2 for w = a + ib, so one step of size 0.1 moves
    # w from 0 to 0.4 + 0.4j, toward the target on both axes.
    np.testing.assert_allclose(new_state.params['w'], 0.4 + 0.4j, rtol=1e-4)
    self.assertEqual(int(new_state.step[0]), 1)

  def test_l1_loss_value(self):
    _, metrics = self._run('l1')
    np.testing.assert_allclose(metrics['loss'], 2.0 * np.sqrt(2.0), rtol=1e-5)


if __name__ == '__main__':
  absltest.main()