# FLDM reference with the published weights (sea lion)
Generation with the FL-VAE and FLDM weights published by the authors
(google-research/mesh_diffusion), on mesh fe973fc8e8c049c6b9ab884137fc9463,
with 1000 diffusion steps (cosine schedule) on an RTX 3090.
## Reproduction
Command run from `fldm/`, with `num_train_samples=100` (a subset of 100
training meshes) and `num_test_samples=16`. Two compatibility fixes were
needed compared to the authors' code, namely `a_max` in
`latents/ring_vae.py` (jax 0.4.25) and reading the `.ply` through `io.BytesIO` in
`latents/io_utils.py` (plyfile 1.0.3).
## Measurements (first order of magnitude, to be redone properly)

* Latent precomputation for 100 meshes, about 3.5 minutes.
* One sample, about 5 min 50, or about 0.35 s per network evaluation.
* The default configuration generates 64 samples, which comes to about 6 h 20 on one GPU.

## Files

* `sample_0.png`, `sample_1.png`, two independent draws.
* `Init_recon_step1.png`, autoencoder reconstruction of a training mesh,
 without diffusion. It gives the quality ceiling of the model.

* `Labels_step1.png`, map of the four label classes that condition the generation.
* `sample_run1.log`, generation log.

## Observations
The samples respect the label distribution (gold, stone, colored patterns).
The sector and cell artifacts are already present in the autoencoder
reconstruction, so they come from the decoder. The samples are less detailed than
the reconstruction, and the gap between the two measures what the diffusion stage loses.