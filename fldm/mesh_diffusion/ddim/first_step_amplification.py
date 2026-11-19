import numpy as np
from mesh_diffusion.sin_im_ddm import diffusion as df
from mesh_diffusion.ddim import samplers

_, _, ab, _, _ = df.get_diffusion_parameters(1000, 'cos')
ab = np.asarray(ab, dtype=np.float64)
for spacing in ('trailing', 'leading'):
  for n in (10, 20, 50, 100, 250, 1000):
    ts = samplers.make_timestep_sequence(1000, n, spacing)
    ab_t, ab_prev = samplers.noise_levels_for_sequence(ab, ts)
    ab_t, ab_prev = float(ab_t[0]), float(ab_prev[0])
    # Coefficient with which an error on the predicted noise reaches z_prev.
    coef = abs(np.sqrt(ab_prev) * np.sqrt(1 - ab_t) / np.sqrt(ab_t)
               - np.sqrt(1 - ab_prev))
    print(f'{spacing:9s} N={n:4d} t0={ts[0]:4d} amplification={coef:12.2f}')