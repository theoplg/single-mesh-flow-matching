# Journal complet du projet single-mesh-flow-matching

Dernière mise à jour le 22 novembre 2026. Ce document rassemble tout ce qui a été fait depuis le début, les problèmes rencontrés avec leurs corrections, les ordres de grandeur de stockage et de temps, et ce qui reste à faire. 

---

## 1. Objectif du projet

Le projet part de FLDM (« Single Mesh Diffusion Models with Field Latents for Texture Generation », CVPR 2024, code `google-research/mesh_diffusion`, en JAX/Flax). FLDM génère de nouvelles textures pour un seul maillage texturé. Le but est de remplacer la partie diffusion par du flow matching en gardant l'équivariance aux isométries, puis de comparer proprement trois échantillonneurs sur les mêmes données.

- DDPM, l'échantillonneur d'origine (1000 pas).
- DDIM, avec un nombre de pas réduit.
- Flow matching, à écrire.

L'asset de travail est la sculpture d'otarie/phoque, identifiant Objaverse `fe973fc8e8c049c6b9ab884137fc9463`.

## 2. Idées techniques à retenir

### FLDM en bref

- Un FL-VAE (gelé, pré-entraîné) encode la texture en un champ de vecteurs tangents, 8 vecteurs complexes par sommet (latents dans C⁸), sur un maillage de 30000 sommets.
- Un petit UNet de field convolutions apprend un DDPM sur ces latents, par asset, sur 500 remaillages différents du même objet (pour ne pas apprendre la connectivité).
- Réglages du code. Prédiction du bruit (epsilon), schedule cosinus dans les trois configs (`ddm_schedule = 'cos'`), T = 1000, perte L1, bruit à l'échelle sigma = √2·moyenne|latents|.
- Le papier mentionne un schedule linéaire, alors que le code d'entraînement des auteurs est en cosinus. On a gardé le code.
- Les latents sont stockés en représentation réelle/imaginaire (…, 2) pour l'échantillonnage et en complexe pour le réseau.

### Ce que le modèle produit réellement

La sortie n'est pas une couleur par sommet. Le décodeur évalue la texture pixel par pixel et produit une image d'atlas UV de 2048×1024. C'est cette image que `train.py` sauvegarde. Une première version de nos explications avait supposé des couleurs par sommet, c'était faux (voir la section 8).

### Flow matching (à implémenter)

- Chemin x_t = (1−t)·x0 + t·x1, avec t = 0 pour le bruit, cible u = x1 − x0, échantillonnage par Euler ou Heun.
- Équivariant aux isométries (une isométrie agit par rotation par sommet), le même UNet est réutilisable.
- Les gains attendus portent surtout sur peu de pas. La comparaison juste est DDIM à nombre égal d'évaluations du réseau (NFE).
- Les poids publiés prédisent le bruit, donc DDPM et DDIM n'ont pas besoin d'entraînement. Seul le flow matching en demandera.

### DDIM

- Estimation du latent propre z0 = (zt − √(1−ᾱ_t)·eps) / √ᾱ_t, puis recombinaison au niveau de bruit suivant.
- `eta = 0` est déterministe, `eta = 1` sur des pas consécutifs retombe sur DDPM (testé).
- Le niveau de bruit du pas t se lit dans `alpha_bar[t-1]`. Le bruit prédit et le bruit frais sont à l'échelle sigma.
- Deux espacements des pas. `leading` (comme `df.get_alphas`, premier pas à 901 pour 10 pas) et `trailing` (premier pas à 1000).

### Pourquoi `trailing` échoue à petit nombre de pas

Avec le schedule cosinus, `alpha_bar[999] ≈ 2,4·10⁻⁹`. Au premier pas, le latent est du bruit presque pur, et une petite erreur de prédiction est multipliée par un grand facteur. Les facteurs d'amplification mesurés pour le premier pas (eta = 0) sont les suivants.

| Pas | `trailing` | `leading` |
|---|---|---|
| 10 | 3148 | 1,01 |
| 20 | 1578 | 1,02 |
| 50 | 631 | 1,05 |
| 100 | 315 | 1,11 |
| 250 | 125 | 1,33 |
| 1000 | 30,6 | 30,6 |

Les images confirment. À 10 pas, `trailing` donne du bruit pur et `leading` une texture propre. À 50 pas, `trailing` redevient lisible mais granuleux. Fichier du résultat prévu dans `results/ddim_first_step_amplification.txt`.

Réserve sur `leading`. À petit nombre de pas, il démarre à t = 901 depuis du bruit pur, alors que la vraie distribution à ce niveau contient un peu de signal (ᾱ ≈ 0,024). Cela peut peser sur la qualité ou la diversité, et seules les métriques sur 64 samples le diront.

## 3. Infrastructure et environnement


### Environnement `fldm` (JAX, TensorFlow)

- Python 3.11.13 chargé avec `module load python/3.11.13`, puis `unset PYTHONPATH` obligatoire.
- Environnement virtuel `~/venvs/fldm`, activé par `~/fldm_env.sh`.
- jax 0.4.25 avec les bibliothèques CUDA 12 installées par pip. Les dépendances épinglées des auteurs sont installées avec `--no-deps`, puis vérifiées par `pip check` et un test d'import.

### Environnement `eval` (PyTorch)

- Environnement virtuel `~/venvs/eval`, activé par `~/fldm_eval_env.sh`.
- Paquets installés. `torch 2.15.0+cu132`, `torchvision`, `pytorch-fid`, `lpips`, `pillow`, `numpy`, `scipy`.
- Deux environnements séparés pour ne pas mélanger JAX et PyTorch.
- Correctif `libbz2` (section 9). Le script d'activation exporte `LD_LIBRARY_PATH=$HOME/lib`.

## 4. Chronologie de ce qui a été fait

### Phase A. Mise en place de l'environnement

1. Création de `~/fldm_env.sh` (le premier essai échouait, aucun environnement n'avait été créé).
2. Installation des paquets, vérification du GPU sur `node40` (une RTX 3090 est bien vue par JAX).
3. Premier recap du projet pour comprendre pourquoi un environnement virtuel et pourquoi `--no-deps`.

### Phase B. Données et points de contrôle

Dans `fldm/mesh_diffusion/data/` se trouvent les archives `fldm.zip` et `flvae.zip` et les dossiers extraits.

```
data/
├── flvae/flvae_checkpoint/        (ckpt-1500, ckpt-1501)
└── meshes/<ID>/
    ├── train/geom_0.tfrecords     (13 Go) + shape_keys.json
    ├── test/geom_0.tfrecords      (2,1 Go) + shape_keys.json
    ├── fldm_sealion/              (ckpt-80, ckpt-81, poids FLDM publiés)
    ├── <ID>_sel.ply               (60197 sommets, 118960 faces, champ quality = labels)
    └── valid_tex.jpg              (202 Ko, texture de référence)
```

Le `_sel.ply` ne contient ni UV ni couleur, seulement la géométrie et les labels. Chaque enregistrement `.tfrecords` contient un maillage remaillé à 30000 sommets, plusieurs niveaux de voisinage, et les liens entre les pixels de la texture et la surface (voir la section 8).

### Phase C. Reproduction de la base FLDM

- Premier échantillonnage avec les poids publiés. Deux correctifs de compatibilité dans le code des auteurs (section 9).
- Lecture des images produites. `Init recon` est la reconstruction du VAE sur la vraie texture et sert de plafond de qualité. `Labels` est la carte de labels. `Sample` est une texture générée. `valid_tex.jpg` est la vraie texture.
- Résultats rangés dans `results/baseline_fldm/` (deux samples, la reconstruction, les labels, le log, un README), commités.
- Le prétraitement des 100 maillages a duré environ 3 min 20, soit l'essentiel du démarrage de 5 min 40 de chaque job.

### Phase D. Benchmark d'entraînement avec SLURM

- Écriture de `fldm/jobs/train_bench.sbatch`, avec trace de la date, du nœud, du job SLURM, du commit git, de `git status`, de `nvidia-smi`, et de l'utilisation GPU toutes les 5 secondes.
- Premier essai (`bench1`) en échec mémoire, second (`bench2`) réussi (voir la section 9).
- Résultats mesurés sur une 3090.

| Mesure | Valeur |
|---|---|
| Vitesse d'entraînement | 0,485 pas/s, soit environ 2,06 s par pas |
| Mémoire d'un pas d'entraînement | 16,6 GiB |
| Utilisation GPU moyenne | 93,9 % |
| Démarrage d'un job | environ 5 min 40 |

- Sur un GPU, un pas correspond à un maillage. `optax.MultiSteps` cumule 16 pas par mise à jour du réseau, et le planning du taux d'apprentissage compte les mises à jour réelles.
- Un job de 2 jours fait donc environ 84 000 pas, soit environ 5 200 mises à jour.
- Les résultats du benchmark n'ont pas été copiés dans `results/bench_train_3090/` (proposé, non confirmé).

### Phase E. Référence DDPM à 1000 pas

- Job `sample_ref.sbatch`, 64 samples, sur `nodemm01`, numéro 1111175, commit `7110268`, limite de temps fixée à 24 h par toi.
- Durée totale 22315 s (6 h 12 min), reconstituée à partir de la date de début et de la date du dernier fichier écrit (`sacct` indisponible).
- L'échantillonnage n'a pas de reprise (l'entraînement en a une, pas l'échantillonnage). Si le job s'interrompt, il faut tout relancer.
- Dossier `mesh_diffusion/runs/ref_ddpm1000/` avec 64 PNG et 64 JPG.

### Phase F. DDIM

Conception décidée ensemble. Tu voulais coder une partie toi-même, donc je t'ai donné le squelette avec les signatures et des docstrings en anglais, les points délicats laissés en TODO.

| Fichier | Rôle |
|---|---|
| `mesh_diffusion/ddim/samplers.py` | `make_timestep_sequence` (TODO 1), `noise_levels_for_sequence` (TODO 2), `ddim_step` (TODO 3), `sample_latents_ddim` (TODO 4 et 5) |
| `mesh_diffusion/ddim/samplers_test.py` | 13 tests (séquences de pas, niveaux de bruit, `ddim_step` contre les fonctions de référence, eta = 0 sans bruit, eta = 1 égal à DDPM, entrées complexes, boucle avec un réseau oracle, reproductibilité, inpainting refusé) |
| `mesh_diffusion/sin_mesh_ddm/train.py` | Branchement minimal (import, un `p_predict` en pmap, une branche `config.sampler == 'ddim'`) |
| `mesh_diffusion/sin_mesh_ddm/configs/default.py` | Quatre options (`sampler`, `num_sampling_steps`, `ddim_eta`, `ddim_spacing`) |
| `jobs/sample_ddim.sbatch` | Balayage du nombre de pas (variables `STEPS_LIST`, `ETA`, `SPACING`, `NSAMP`) |

- Les tests ont été validés de mon côté contre une implémentation de référence privée, avec des mutations volontaires détectées, sous jax 0.4.38 et pas sous 0.4.25.
- Sur ta remarque, le code DDIM a été sorti du dossier des auteurs vers `mesh_diffusion/ddim/`, pour que ton apport se voie. Le seul fichier de Google qui reste modifié est `train.py` (une quinzaine de lignes) et `default.py` (4 options).
- Le script de branchement est idempotent et il a fallu corriger l'import (`ddim` et non `sin_mesh_ddm`).

Tests sur le GPU (2 samples) puis balayages complets sur 64 samples.

| Pas | Temps `leading` (s) | Temps `trailing` (s) |
|---|---|---|
| 10 | 556 | 554 |
| 20 | 744 | 754 |
| 50 | 1384 | 1395 |
| 100 | 2449 | 2482 |
| 250 | 5669 | 5734 |
| 1000 (DDPM) | 22315 | — |

Une droite sur ces points donne environ 340 s de démarrage fixe et 21,3 s par pas pour 64 samples, soit environ 0,33 s par évaluation du réseau, cohérent avec la mesure du benchmark (0,34 s). Hors démarrage, le DDIM à 10 pas est environ 100 fois plus rapide que DDPM. Démarrage compris, il n'est que 40 fois plus rapide.

### Phase G. Comprendre et rendre les sorties

- Les images produites sont des atlas de texture, pas des vues du phoque. Le code les sauvegarde en double (le `.jpg` est un PNG renommé).
- Le fichier `shape_keys.json` et les enregistrements donnent, pour 819148 pixels utiles de la texture, le triangle qui les contient et leurs coordonnées barycentriques. On en déduit la position 3D de chaque pixel, donc on peut poser la texture sur la surface sans maillage UV.
- `render_views.py`. Planche de 8 vues pour regarder une ou deux textures, avec validation sur la vraie texture `valid_tex.jpg`. Elle a servi à choisir l'axe vertical, `--up z`.
- `render_set.py`. Rendu d'évaluation avec 24 caméras fixes (24 azimuts, élévation 45°, image de 512 px, axe z, splat de 1 pixel). La carte de visibilité de chaque vue est calculée une fois, puis chaque texture est colorée par indexation. Sortie `view_XX/ref.png` et `view_XX/<n>.png`, plus `render_config.json` (paramètres et commit).
- Le rendu du sample 0 de DDPM montre un phoque reconnaissable, à l'endroit, avec les bonnes zones colorées, même si le rocher est plus flou que la vraie texture.

### Phase H. Métriques

Protocole du papier (section 6.2). 64 textures par asset, rendues sous 24 vues. SIFID entre le rendu de chaque sample et celui de la texture d'origine sous la même vue (fidélité). LPIPS entre les rendus de deux samples sous la même vue (diversité). Valeurs du papier affichées avec SIFID × 10⁵ et LPIPS × 10 (FLDM 3,27 et 1,15 sur 10 assets). Sin3DM, dont le protocole est repris, rend 8 vues à 45° d'élévation dans Blender.

- `metrics.py`. SIFID sur les features du premier max pooling d'InceptionV3 (64 canaux, via `pytorch-fid`, sans redimensionnement), moyenne et covariance sur les positions spatiales d'une image, puis distance de Fréchet. LPIPS avec le réseau AlexNet, moyenne sur les paires non ordonnées. Sortie `metrics.json` avec les valeurs brutes et celles du papier.
- `summarize.py`. Tableau (markdown et CSV) rassemblant les `metrics.json`.
- `eval_runs.sbatch`. Enchaîne le rendu (environnement `fldm`) puis les métriques (environnement `eval`) pour une liste de runs, saute ceux déjà évalués (`FORCE=1` pour recalculer), et continue si un run plante.
- `render_set.sbatch` et `metrics.sbatch` pour les étapes isolées.

Premier test de bout en bout sur 4 samples de la référence DDPM (sans valeur statistique).

| Mesure | Valeur |
|---|---|
| SIFID(ref, ref), contrôle | 2,3·10⁻¹³ |
| SIFID moyen | 0,851 (écart-type 0,043 entre samples, de 0,46 à 1,36 selon la vue) |
| LPIPS moyen | 0,0804 (soit 0,80 une fois multiplié par 10) |

Le LPIPS est dans l'ordre de grandeur du papier. Le SIFID n'est pas comparable à celui du papier (rendu différent, un seul asset, normalisation inconnue). Seule la comparaison entre méthodes avec notre pipeline a du sens.

## 5. État actuel

- Les 10 runs DDIM (5 nombres de pas × 2 espacements) et la référence DDPM sont complets, 64 PNG chacun.
- La prochaine étape est l'évaluation de ces 11 runs avec `eval_runs.sbatch`, puis `summarize.py`.
- Le commit des fichiers DDIM, rendu et métriques, et du script d'amplification, a été recommandé. Je ne sais pas s'il est fait.

## 6. Problèmes rencontrés et corrections

| Problème | Cause | Correction |
|---|---|---|
| `ModuleNotFoundError: tensorflow` ou `absl` dans un shell neuf | Environnement non activé, `~/fldm_env.sh` inexistant | Création du script d'activation |
| `jnp.clip() got an unexpected keyword argument 'max'` | jax 0.4.25 attend `a_max` | `ring_vae.py` ligne 174, `max=3.0` devient `a_max=3.0` (commité) |
| `'GFile' object has no attribute 'readable'` | plyfile 1.0.3 | `io_utils.py`, ajout de `import io` et lecture par `io.BytesIO(f.read())` (commité) |
| Bloc `srun` collé sur la passerelle (CpuDevice, déconnexion) | La commande a tourné sur `gpu-gw` et non sur un nœud | Se reconnecter, coller les blocs séparément, vérifier que le prompt affiche `nodeXX` |
| `sbatch: Rejected [3090]: max 4 CPUs/GPU` | Règle de la partition | `--cpus-per-task=4` |
| `cosine_decay_schedule requires positive decay_steps, got -700` | `warmup_steps=1000` avec un benchmark de 300 pas | `--config.warmup_steps=50` pour le benchmark (garder 1000 pour un vrai entraînement) |
| `RESOURCE_EXHAUSTED` sur une allocation de 16,15 GiB (bench1) | Le flag `XLA_PYTHON_CLIENT_PREALLOCATE=false` que j'avais proposé provoque une fragmentation de la mémoire | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`, relance en `bench2` réussie |
| Extraction des images TensorBoard sans résultat | Les images sont dans les tenseurs, pas dans les événements d'image | Lecture des tags `tensors` (`string_val[2]`) |
| Bloc `cat > fichier <<'EOF'` collé avec la commande suivante | Le `EOF` s'est collé au `cd` suivant (`EOFcd`), le shell n'a pas fermé le fichier | Supprimer la fin parasite (`sed -i '/^EOFcd/,$d'`), coller les blocs un par un ou utiliser l'éditeur |
| Deux commandes `sbatch` collées sur une ligne | La deuxième est lue comme arguments de la première | Les lancer sur deux lignes, annuler avec `scancel <numéro>` ou `scancel --name=fldm_ddim` |
| `sacct` indisponible | Base de comptabilité du cluster inactive | Durée reconstituée avec la date de début de `meta.txt` et la date du dernier fichier écrit |
| `ImportError: libbz2.so.1.0` à l'import de `torchvision` | Le Python du module cherche `libbz2.so.1.0`, le nœud n'a que `libbz2.so.1` | Lien `~/lib/libbz2.so.1.0` vers `/usr/lib64/libbz2.so.1` et `LD_LIBRARY_PATH=$HOME/lib` dans `~/fldm_eval_env.sh` |
| DDIM `trailing` à 10 pas donne du bruit | Amplification d'erreur au premier pas (ᾱ presque nul) | Pas un bug, comparaison avec `leading`, balayage dans les deux espacements |
| Sandbox sans jaxlib 0.4.30 (Python 3.13) | Version de Python | Validation privée avec jax 0.4.38, numpy 2.0.2 |
| Patch du branchement collé avec des espaces mal copiés | Indentation sensible | Script Python idempotent avec assertions d'unicité |

### Erreurs de ma part, corrigées

- J'avais affirmé que le décodeur donnait une couleur par sommet. Le code décode des pixels de texture, et la proposition de sauvegarder des `.npy` de couleurs par sommet a été abandonnée. Aucun job n'a été relancé pour cela.
- Dans `render_views.py`, le signe de l'élévation était inversé (une élévation positive donnait une vue par en dessous). `render_set.py` utilise le bon signe, et la correction de deux lignes pour `render_views.py` t'a été donnée.
- J'avais d'abord placé le code DDIM dans le dossier des auteurs. Corrigé suite à ta remarque.

## 7. Stockage et ressources

Aucune erreur de quota n'a été rencontrée. Les points ci-dessous sont des risques ou des mesures.

### Tailles connues

| Élément | Taille |
|---|---|
| Quota du dossier personnel | 100 GiB |
| `train/geom_0.tfrecords` (100 maillages) | 13 Go |
| `test/geom_0.tfrecords` (16 maillages) | 2,1 Go |
| Un enregistrement de géométrie | environ 130 Mo |
| Estimation pour 500 maillages d'entraînement | 65 à 67 Go, soit les deux tiers du quota |
| Samples d'un run (64 PNG + 64 JPG) | 61 Mo |
| Rendus d'un run (24 vues × 65 images) | 167 Mo |
| Images de la base FLDM versionnées | environ 740 Ko chacune |
| Mémoire GPU nécessaire à un pas d'entraînement | 16,6 GiB sur 24 |

- Le `ls -lh` de `train` indique un fichier de 13 Go mais « total 6,8G » en blocs, ce qui peut venir de la façon dont le système de fichiers compte l'espace. À confirmer avec `du -sh`.
- Pour les 11 runs évalués, il faut compter environ 0,7 Go de samples et environ 1,8 Go de rendus. Les environnements `~/venvs/fldm` et surtout `~/venvs/eval` (PyTorch et ses bibliothèques CUDA) occupent chacun plusieurs Go, non mesurés.
- Chaque dossier de run contient aussi un dossier `checkpoints-0`, un événement TensorBoard et un `pip_freeze.txt`, de tailles non mesurées.

### Précautions mises en place

- `.gitignore` à la racine pour `fldm/mesh_diffusion/data/`, `runs/`, `samples/`, `workdir/` et `fldm/jobs/logs/`, afin de ne pas versionner les gros fichiers (on ajoute aussi `__pycache__/` et `.pytest_cache/`).
- Les résultats utiles sont copiés à la main vers `results/` et commités (images de la base, tableaux).

### Mesures à faire pour savoir où on en est

```bash
quota -s 2>/dev/null; df -h ~
du -sh ~/venvs/* ~/single-mesh-flow-matching/fldm/mesh_diffusion/data ~/single-mesh-flow-matching/fldm/mesh_diffusion/runs
du -sh ~/single-mesh-flow-matching/fldm/mesh_diffusion/runs/*
```

### Points d'attention pour la suite

- Passer à 500 maillages d'entraînement est possible mais ferait monter les données à environ 65 à 67 Go, avec les environnements et les runs on s'approcherait vite des 100 GiB. Les données doivent rester sur le cluster, et il faut éviter de dupliquer.
- Chaque job recalcule les latents au démarrage (3 à 5 minutes). Un cache des latents économiserait du temps, mais coûterait de l'espace.
- Le dépôt local sur ton Mac se trouvait dans iCloud. Déplacer le dépôt hors d'iCloud avait été noté comme bonne pratique (tu travailles maintenant par VS Code en SSH, à confirmer).

## 8. Découvertes sur le code des auteurs

- Les trois configs utilisent le schedule cosinus, alors que le papier parle de schedule linéaire.
- `train.py` (lignes 1263 à 1267) écrit le même tableau deux fois avec `mio.save_png`, sous un nom `.png` et un nom `.jpg`. Les JPG sont donc de simples copies.
- Chaque enregistrement de géométrie contient les champs `nodes` (30000×3), `pix_tris`, `pix_bary`, `I`, `J`, `valid_ind` (819148 entrées) et `tex_dim`, en plus des niveaux de voisinage (`hei_*`, `bases_*`, `logs_*`, `xport_*`, `unpool_*`) et des anneaux (`ring_logs`, `ring_vals`). Les tenseurs sont sérialisés avec `tf.io.serialize_tensor`.
- Le champ `pix_vals` n'a pas été utilisé. Son rôle (peut-être la couleur de référence en entiers) n'est pas vérifié.
- L'échantillonnage utilise une graine fixe (clé 4294967295, 4294967295) quand `config.seed` vaut `None`. Avec `eta = 0`, les runs DDIM partent donc des mêmes tirages de bruit initiaux, ce qui rend leurs images comparables entre elles.

## 9. Décisions prises

- Rester dans le dépôt des auteurs pour le minimum de code modifié, tout le reste dans des dossiers à toi (`ddim/`, `evaluation/`, plus tard `flow_matching/`).
- Comparer DDIM et flow matching à nombre égal d'évaluations du réseau, et mesurer aussi le temps réel.
- Garder `leading` et `trailing` comme ablation plutôt que de choisir l'un des deux d'avance.
- Évaluer sur les 64 samples × 24 vues avec notre propre rendu, sans chercher à reproduire les chiffres du papier.
- Utiliser des jobs SLURM tracés (`meta.txt`) plutôt que des commandes tapées dans le terminal.

## 10. Ce qui reste à faire

1. Évaluer les 11 runs (`eval_runs.sbatch`) et produire le tableau (`summarize.py`).
2. Commiter les fichiers DDIM, rendu, métriques, résultats.
3. Lire le tableau. Voir si `leading` à 10 pas tient face à DDPM, à quel nombre de pas `trailing` rattrape, et si les écarts dépassent le bruit entre samples.
4. Écrire le flow matching (perte, échantillonneur, adaptation de l'inpainting) en réutilisant l'interface `predict_fn` de `samplers.py`.
5. Définir un budget d'entraînement équitable, en nombre de mises à jour et non de pas, et se demander s'il faut aussi un DDPM entraîné au même budget. Démarrage à chaud ou à froid.
6. Vérifier le comportement du planning du taux d'apprentissage avec `MultiSteps` sur un GPU (hypothèse non confirmée) et l'entraînement multi-GPU (taille de lot 16 divisée par le nombre de GPU).
7. Décider entre 100 et 500 maillages d'entraînement selon l'espace disponible, et envisager un cache des latents.
8. Optionnel. Calculer le SIFID de la reconstruction du VAE (plafond) et celui d'une texture volontairement mauvaise, pour donner une échelle aux valeurs.
9. Optionnel. Copier les résultats du benchmark dans `results/bench_train_3090/`.

## 11. Incertitudes

- La configuration exacte du SIFID du papier (couche, normalisation). On utilise 64 canaux, sans redimensionnement, d'après l'implémentation de SinGAN.
- Les 24 caméras précises du papier. On a choisi 24 azimuts à 45° d'élévation.
- Le rendu est un nuage de points colorés, sans éclairage, pas un rendu Blender.

## 12. Annexe

### Commandes utiles

```bash
# Activer l'environnement JAX / TensorFlow, ou celui de PyTorch
source ~/fldm_env.sh
source ~/fldm_eval_env.sh

# Tests de l'échantillonneur DDIM
JAX_PLATFORMS=cpu python -m mesh_diffusion.ddim.samplers_test

# Balayage DDIM (deux espacements)
sbatch jobs/sample_ddim.sbatch
sbatch --export=ALL,SPACING=leading jobs/sample_ddim.sbatch

# Évaluation de plusieurs runs, puis tableau
sbatch --export=ALL,RUNS="ref_ddpm1000 ddim_n10_eta0.0_leading ..." jobs/eval_runs.sbatch
python mesh_diffusion/evaluation/summarize.py --runs_dir mesh_diffusion/runs --out ../results/eval_summary.csv

# Suivre ou annuler des jobs
squeue -u $USER
scancel --name=fldm_ddim
```

### Fichiers créés

```
fldm/jobs/train_bench.sbatch        fldm/jobs/sample_ref.sbatch
fldm/jobs/sample_ddim.sbatch        fldm/jobs/render_set.sbatch
fldm/jobs/metrics.sbatch            fldm/jobs/eval_runs.sbatch
fldm/mesh_diffusion/ddim/           samplers.py, samplers_test.py, first_step_amplification.py
fldm/mesh_diffusion/evaluation/     render_views.py, render_set.py, metrics.py, summarize.py
results/baseline_fldm/              images et log de la base
results/ddim_first_step_amplification.txt
```

### Repères de traçabilité

- Job de la référence DDPM, numéro 1111175, nœud `nodemm01`, commit `7110268`, 22315 s.
- Chaque run contient `meta.txt` (date, nœud, job, commit, `git status`, `nvidia-smi`, `wall_seconds`).
- Chaque dossier de rendu contient `render_config.json`.
- Dossier d'un run. `mesh_diffusion/runs/<nom>/{samples, renders, meta.txt, metrics.json, run.log}`.
