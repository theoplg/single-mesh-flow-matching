# coding=utf-8
"""Computes SIFID (fidelity) and LPIPS (diversity) on rendered views.

Expects the layout produced by `render_set.py`, with `ref.png` and one image per
sample in every `view_XX` directory.

SIFID is the Frechet distance between the statistics of the Inception features
of two single images, computed over spatial positions, between the reference
view and each sample view. LPIPS is the mean distance between pairs of sample
views. Both are averaged over views, as in the evaluation of Sin3DM and FLDM.

Example, from the `fldm/` directory:
  python mesh_diffusion/evaluation/metrics.py \
    --render_dir mesh_diffusion/runs/ref_ddpm1000/renders \
    --out mesh_diffusion/runs/ref_ddpm1000/metrics.json
"""
import argparse
import glob
import json
import os
import re

import lpips
import numpy as np
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import torch


def sorted_naturally(paths):
  """Sorts paths so that 2.png comes before 10.png."""
  def key(path):
    parts = re.split(r'(\d+)', os.path.basename(path))
    return [int(p) if p.isdigit() else p for p in parts]
  return sorted(paths, key=key)


def load_image(path, device):
  """Loads an image as a float tensor (1, 3, H, W) with values in [0, 1]."""
  array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
  return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


class Sifid:
  """Single Image Frechet Inception Distance."""

  def __init__(self, device, block=0):
    """Builds the feature extractor.

    Args:
      device: Torch device.
      block: Index of the Inception block that gives the features. 0 is the first
        max pooling (64 channels), 1 the second (192), 2 gives 768 and 3 gives
        2048 channels.
    """
    self.model = InceptionV3(
        [block], resize_input=False, normalize_input=True).to(device).eval()

  @torch.no_grad()
  def stats(self, image):
    """Computes mean and covariance of the features of one image.

    Args:
      image: Tensor (1, 3, H, W) with values in [0, 1].

    Returns:
      A tuple (mean, covariance) of numpy arrays of shapes (C,) and (C, C).
    """
    feat = self.model(image)[0]  # (1, C, h, w)
    act = feat.squeeze(0).flatten(1).T.double().cpu().numpy()  # (h * w, C)
    return act.mean(axis=0), np.cov(act, rowvar=False)

  @staticmethod
  def distance(stats_a, stats_b):
    """Frechet distance between two (mean, covariance) pairs."""
    return float(calculate_frechet_distance(*stats_a, *stats_b))


@torch.no_grad()
def mean_pairwise_lpips(images, loss_fn, chunk=64):
  """Mean LPIPS over all unordered pairs of images.

  Args:
    images: Tensor (n, 3, H, W) with values in [0, 1].
    loss_fn: An `lpips.LPIPS` module.
    chunk: Number of pairs evaluated at once.

  Returns:
    The mean distance as a float.
  """
  n = images.shape[0]
  ii, jj = torch.triu_indices(n, n, offset=1, device=images.device)
  scaled = images * 2.0 - 1.0
  values = []
  for start in range(0, ii.numel(), chunk):
    a = scaled[ii[start:start + chunk]]
    b = scaled[jj[start:start + chunk]]
    values.append(loss_fn(a, b).flatten())
  return float(torch.cat(values).mean())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--render_dir', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--sifid_block', type=int, default=0)
  parser.add_argument('--max_samples', type=int, default=0,
                      help='Use only the first samples, 0 means all.')
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  sifid = Sifid(device, args.sifid_block)
  lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

  view_dirs = sorted(glob.glob(os.path.join(args.render_dir, 'view_*')))
  if not view_dirs:
    raise SystemExit(f'No view_* directory in {args.render_dir}')

  sifid_scores = []  # (views, samples)
  lpips_scores = []  # (views,)
  for view_dir in view_dirs:
    ref = load_image(os.path.join(view_dir, 'ref.png'), device)
    paths = [p for p in sorted_naturally(glob.glob(
        os.path.join(view_dir, '*.png'))) if os.path.basename(p) != 'ref.png']
    if args.max_samples:
      paths = paths[:args.max_samples]
    images = torch.cat([load_image(p, device) for p in paths])

    ref_stats = sifid.stats(ref)
    if not sifid_scores:
      print('sanity SIFID(ref, ref) =', sifid.distance(ref_stats, ref_stats))
    sifid_scores.append([
        sifid.distance(ref_stats, sifid.stats(images[i:i + 1]))
        for i in range(len(paths))])
    lpips_scores.append(mean_pairwise_lpips(images, lpips_fn))
    print(f'{os.path.basename(view_dir)}: sifid={np.mean(sifid_scores[-1]):.6f} '
          f'lpips={lpips_scores[-1]:.4f}', flush=True)

  sifid_scores = np.asarray(sifid_scores)
  lpips_scores = np.asarray(lpips_scores)
  per_sample = sifid_scores.mean(axis=0)
  result = {
      'n_views': int(sifid_scores.shape[0]),
      'n_samples': int(sifid_scores.shape[1]),
      'sifid_block': args.sifid_block,
      'sifid_mean': float(sifid_scores.mean()),
      'sifid_mean_x1e5': float(sifid_scores.mean() * 1e5),
      'sifid_std_over_samples': float(per_sample.std()),
      'lpips_mean': float(lpips_scores.mean()),
      'lpips_mean_x10': float(lpips_scores.mean() * 10),
      'lpips_std_over_views': float(lpips_scores.std()),
      'sifid_per_view': sifid_scores.mean(axis=1).tolist(),
      'lpips_per_view': lpips_scores.tolist(),
  }
  with open(args.out, 'w') as f:
    json.dump(result, f, indent=2)
  print(json.dumps({k: v for k, v in result.items() if 'per_view' not in k},
                   indent=2))


if __name__ == '__main__':
  main()