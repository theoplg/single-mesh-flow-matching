# Référence FLDM avec les poids publiés (lion de mer)

Génération avec les poids du FL-VAE et du FLDM publiés par les auteurs
(google-research/mesh_diffusion), sur le maillage fe973fc8e8c049c6b9ab884137fc9463,
avec 1000 pas de diffusion (planning cosinus) sur une RTX 3090.

## Reproduction

Commande lancée depuis `fldm/`, avec `num_train_samples=100` (sous-ensemble de 100
maillages d'entraînement) et `num_test_samples=16`. Deux corrections de compatibilité
ont été nécessaires par rapport au code des auteurs, à savoir `a_max` dans
`latents/ring_vae.py` (jax 0.4.25) et la lecture du `.ply` via `io.BytesIO` dans
`latents/io_utils.py` (plyfile 1.0.3).

## Mesures (premier ordre de grandeur, à refaire proprement)

- Précalcul des latents pour 100 maillages, environ 3,5 minutes.
- Un échantillon, environ 5 min 50, soit environ 0,35 s par évaluation du réseau.
- La configuration par défaut génère 64 échantillons, soit environ 6 h 20 sur une GPU.

## Fichiers

- `sample_0.png`, `sample_1.png`, deux tirages indépendants.
- `Init_recon_step1.png`, reconstruction par l'auto-encodeur d'un maillage
  d'entraînement, sans diffusion. Elle donne le plafond de qualité du modèle.
- `Labels_step1.png`, carte des quatre classes d'étiquettes qui conditionne la génération.
- `sample_run1.log`, journal de la génération.

## Observations

Les échantillons respectent la répartition des étiquettes (or, pierre, motifs colorés).
Les artefacts en secteurs et en cellules sont déjà présents dans la reconstruction de
l'auto-encodeur, ils viennent donc du décodeur. Les échantillons sont moins détaillés
que la reconstruction, l'écart entre les deux mesure ce que perd l'étage de diffusion.
