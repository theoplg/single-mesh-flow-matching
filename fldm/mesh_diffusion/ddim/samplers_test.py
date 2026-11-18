# coding=utf-8
"""Unit tests for `samplers`. They run on CPU in a few seconds.

Run from the `fldm/` directory with
  python -m mesh_diffusion.sin_mesh_ddm.samplers_test
"""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from mesh_diffusion.sin_im_ddm import diffusion as df
from mesh_diffusion.ddim import samplers

T = 1000


class TimestepSequenceTest(absltest.TestCase):

  def test_trailing_starts_at_full_noise(self):
    ts = samplers.make_timestep_sequence(T, 10, 'trailing')
    np.testing.assert_array_equal(ts, np.arange(1000, 0, -100))

  def test_leading_matches_get_alphas(self):
    for n in (10, 20, 50, 100):
      ts = samplers.make_timestep_sequence(T, n, 'leading')
      _, _, tlist = df.get_alphas(T, tau=n, schedule='cos')
      np.testing.assert_array_equal(ts, np.asarray(tlist)[::-1])

  def test_full_length_gives_all_timesteps(self):
    for spacing in ('leading', 'trailing'):
      ts = samplers.make_timestep_sequence(T, T, spacing)
      np.testing.assert_array_equal(ts, np.arange(T, 0, -1))

  def test_basic_properties(self):
    for spacing in ('leading', 'trailing'):
      for n in (1, 7, 25, 250):
        ts = samplers.make_timestep_sequence(T, n, spacing)
        self.assertLen(ts, n)
        self.assertTrue(np.all(np.diff(ts) < 0))
        self.assertGreaterEqual(ts.min(), 1)
        self.assertLessEqual(ts.max(), T)

  def test_invalid_arguments(self):
    with self.assertRaises(ValueError):
      samplers.make_timestep_sequence(T, 0)
    with self.assertRaises(ValueError):
      samplers.make_timestep_sequence(T, T + 1)
    with self.assertRaises(ValueError):
      samplers.make_timestep_sequence(T, 10, 'unknown')


class NoiseLevelsTest(absltest.TestCase):

  def test_levels_and_successors(self):
    _, _, alpha_bar, _, _ = df.get_diffusion_parameters(T, 'cos')
    ts = np.array([1000, 600, 200, 1])
    ab_t, ab_prev = samplers.noise_levels_for_sequence(alpha_bar, ts)
    np.testing.assert_allclose(ab_t, np.asarray(alpha_bar)[ts - 1])
    np.testing.assert_allclose(ab_prev[:-1], ab_t[1:])
    self.assertEqual(float(ab_prev[-1]), 1.0)
    self.assertEqual(ab_t.shape, (4,))
    self.assertEqual(ab_prev.shape, (4,))


class DdimStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _, _, self.alpha_bar, _, _ = df.get_diffusion_parameters(T, 'cos')
    key_z, key_e, key_n = jax.random.split(jax.random.PRNGKey(0), 3)
    self.zt = jax.random.normal(key_z, (6, 4))
    self.eps = jax.random.normal(key_e, (6, 4))
    self.noise = jax.random.normal(key_n, (6, 4))

  def test_matches_reference_helper(self):
    ab_t, ab_prev = self.alpha_bar[599], self.alpha_bar[499]
    z0 = df.predict_z0(ab_t, self.zt, self.eps)
    for eta in (0.0, 0.5, 1.0):
      expected = df.alpha_predict_prev_from_z0(
          ab_t, ab_prev, self.zt, z0, self.noise, eta)
      actual = samplers.ddim_step(
          ab_t, ab_prev, self.zt, self.eps, self.noise, eta)
      np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  def test_eta_zero_ignores_noise(self):
    ab_t, ab_prev = self.alpha_bar[599], self.alpha_bar[499]
    a = samplers.ddim_step(ab_t, ab_prev, self.zt, self.eps, self.noise, 0.0)
    b = samplers.ddim_step(ab_t, ab_prev, self.zt, self.eps, 7.0 * self.noise,
                           0.0)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

  def test_eta_one_on_consecutive_steps_is_ddpm(self):
    beta, alpha, alpha_bar, alpha_bar_prev, ln_var = df.get_diffusion_parameters(
        T, 'cos')
    for t in (1, 2, 300, 999, 1000):
      expected = df.predict_prev(
          beta[t - 1], alpha[t - 1], alpha_bar[t - 1], alpha_bar_prev[t - 1],
          ln_var[t - 1], self.zt, self.eps, self.noise)
      actual = samplers.ddim_step(
          alpha_bar[t - 1], alpha_bar_prev[t - 1], self.zt, self.eps,
          self.noise, 1.0)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

  def test_complex_inputs(self):
    ab_t, ab_prev = self.alpha_bar[599], self.alpha_bar[499]
    zt = jax.lax.complex(self.zt, self.eps)
    eps = jax.lax.complex(self.eps, self.noise)
    noise = jax.lax.complex(self.noise, self.zt)
    out = samplers.ddim_step(ab_t, ab_prev, zt, eps, noise, 0.5)
    self.assertTrue(jnp.iscomplexobj(out))
    self.assertEqual(out.shape, zt.shape)


class SampleLatentsDdimTest(absltest.TestCase):
  """Checks the sampling loop with a network that knows the clean latent."""

  def _run(self, num_steps, eta, spacing):
    num_verts, latent_dim = 5, 8
    _, _, alpha_bar, _, _ = df.get_diffusion_parameters(T, 'cos')
    key = jax.random.PRNGKey(1)
    z0 = jax.lax.complex(
        jax.random.normal(jax.random.PRNGKey(2), (num_verts, latent_dim)),
        jax.random.normal(jax.random.PRNGKey(3), (num_verts, latent_dim)))

    def oracle(params, geom_data, zt, t):
      del params, geom_data
      ab = alpha_bar[t.astype(jnp.int32) - 1]
      return (zt - jnp.sqrt(ab) * z0) / jnp.sqrt(1.0 - ab)

    predict_fn = jax.pmap(oracle, axis_name='batch')
    latents, new_key = samplers.sample_latents_ddim(
        predict_fn=predict_fn,
        geom_data=jnp.zeros((1,)),
        params=jnp.zeros((1,)),
        key=key,
        shape=(1, num_verts, latent_dim, 2),
        timestep=T,
        sigma=0.7,
        schedule='cos',
        num_sampling_steps=num_steps,
        eta=eta,
        spacing=spacing)
    return z0, latents, key, new_key

  def test_oracle_network_recovers_clean_latent(self):
    for spacing in ('leading', 'trailing'):
      for num_steps in (10, 50):
        for eta in (0.0, 1.0):
          z0, latents, _, _ = self._run(num_steps, eta, spacing)
          self.assertEqual(latents.shape, (1,) + z0.shape)
          self.assertTrue(jnp.iscomplexobj(latents))
          np.testing.assert_allclose(latents[0], z0, atol=1e-3, rtol=1e-3)

  def test_deterministic_sampler_is_reproducible(self):
    _, a, _, _ = self._run(10, 0.0, 'trailing')
    _, b, _, _ = self._run(10, 0.0, 'trailing')
    np.testing.assert_array_equal(a, b)

  def test_inpainting_is_rejected(self):
    with self.assertRaises(NotImplementedError):
      samplers.sample_latents_ddim(
          predict_fn=None, geom_data=None, params=None,
          key=jax.random.PRNGKey(0), shape=(1, 2, 8, 2), timestep=T,
          inpaint_data=(None, None))


if __name__ == '__main__':
  absltest.main()
