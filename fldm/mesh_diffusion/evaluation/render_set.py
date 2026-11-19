# coding=utf-8
"""Renders a reference texture and generated textures under fixed cameras.

All textures are rendered with exactly the same cameras, so that renderings of
the reference and of the samples can be compared view by view. The output
layout is
  <out_dir>/view_00/ref.png
  <out_dir>/view_00/<sample name>.png
  ...
  <out_dir>/render_config.json

Example, from the `fldm/` directory:
  python mesh_diffusion/evaluation/render_set.py \
    --tfrecord mesh_diffusion/data/meshes/<ID>/test/geom_0.tfrecords \
    --reference mesh_diffusion/data/meshes/<ID>/valid_tex.jpg \
    --samples_dir mesh_diffusion/runs/ref_ddpm1000/samples \
    --out_dir mesh_diffusion/runs/ref_ddpm1000/renders
"""
import argparse
import glob
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image

from render_views import read_record, texel_colors, texel_points


def sorted_naturally(paths):
  """Sorts paths so that 2.png comes before 10.png."""
  def key(path):
    parts = re.split(r'(\d+)', os.path.basename(path))
    return [int(p) if p.isdigit() else p for p in parts]
  return sorted(paths, key=key)


def project(points, azimuth, elevation, up):
  """Computes screen coordinates and depth for a camera above the object.

  The projection is orthographic. A positive elevation puts the camera above the
  object, looking down. Only the 'z' and 'x' up axes give a non-mirrored image
  when the model coordinates are right-handed.

  Args:
    points: Float array (n, 3), centered and scaled to the unit ball.
    azimuth: Rotation around the vertical axis, in radians.
    elevation: Elevation of the camera above the horizontal plane, in radians.
    up: Index (0, 1 or 2) of the coordinate that points up in the model.

  Returns:
    A tuple (x, y, depth) of arrays of shape (n,). x points right and y points up
    on the screen. A smaller depth means closer to the camera.
  """
  horiz = [i for i in range(3) if i != up]
  a, b, v = points[:, horiz[0]], points[:, horiz[1]], points[:, up]
  x = np.cos(azimuth) * a - np.sin(azimuth) * b
  depth = np.sin(azimuth) * a + np.cos(azimuth) * b
  y = np.cos(elevation) * v + np.sin(elevation) * depth
  depth = np.cos(elevation) * depth - np.sin(elevation) * v
  return x, y, depth


def visibility_map(x, y, depth, size, splat):
  """Finds, for every pixel, the nearest point that projects onto it.

  Each point covers a square of (2 * splat + 1) pixels. The result does not
  depend on colors, so it can be reused for any texture.

  Args:
    x: Screen abscissa of each point, in [-1, 1].
    y: Screen ordinate of each point, in [-1, 1].
    depth: Depth of each point, smaller is closer.
    size: Side of the square image in pixels.
    splat: Half-width of the square drawn for each point.

  Returns:
    An int array of shape (size, size) with the index of the visible point at
    each pixel, or -1 for the background.
  """
  scale = 0.45 * size
  px = np.round(x * scale + size / 2).astype(np.int64)
  py = np.round(size / 2 - y * scale).astype(np.int64)
  offsets = np.arange(-splat, splat + 1)
  dx, dy = np.meshgrid(offsets, offsets)
  dx, dy = dx.ravel(), dy.ravel()
  k = dx.size
  xi = (px[:, None] + dx[None, :]).ravel()
  yi = (py[:, None] + dy[None, :]).ravel()
  di = np.repeat(depth, k)
  idx = np.repeat(np.arange(len(x)), k)
  ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
  xi, yi, di, idx = xi[ok], yi[ok], di[ok], idx[ok]

  flat = yi * size + xi
  order = np.lexsort((di, flat))  # by pixel, then nearest first
  flat, idx = flat[order], idx[order]
  first = np.concatenate([[True], flat[1:] != flat[:-1]])
  vis = np.full(size * size, -1, np.int64)
  vis[flat[first]] = idx[first]
  return vis.reshape(size, size)


def shade(colors, vis):
  """Builds an image from per-point colors and a visibility map.

  Args:
    colors: Uint8 array (n, 3), one color per point.
    vis: Visibility map returned by `visibility_map`.

  Returns:
    A uint8 image of shape (size, size, 3) with a black background.
  """
  image = np.zeros(vis.shape + (3,), np.uint8)
  mask = vis >= 0
  image[mask] = colors[vis[mask]]
  return image


def git_commit():
  """Returns the current git commit hash, or 'unknown'."""
  try:
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()
  except Exception:  # pylint: disable=broad-except
    return 'unknown'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tfrecord', required=True)
  parser.add_argument('--shape_keys', default=None)
  parser.add_argument('--index', type=int, default=0)
  parser.add_argument('--reference', required=True,
                      help='Texture image of the input object.')
  parser.add_argument('--samples_dir', required=True,
                      help='Directory with the generated textures (*.png).')
  parser.add_argument('--out_dir', required=True)
  parser.add_argument('--views', type=int, default=24)
  parser.add_argument('--elevation_deg', type=float, default=45.0)
  parser.add_argument('--size', type=int, default=512)
  parser.add_argument('--up', choices=['x', 'y', 'z'], default='z')
  parser.add_argument('--splat', type=int, default=1)
  parser.add_argument('--limit', type=int, default=0,
                      help='Render only the first samples, 0 means all.')
  args = parser.parse_args()

  shape_keys = args.shape_keys or os.path.join(
      os.path.dirname(args.tfrecord), 'shape_keys.json')
  sample_paths = sorted_naturally(
      glob.glob(os.path.join(args.samples_dir, '*.png')))
  if args.limit:
    sample_paths = sample_paths[:args.limit]
  if not sample_paths:
    raise SystemExit(f'No png found in {args.samples_dir}')
  names = [os.path.splitext(os.path.basename(p))[0] for p in sample_paths]
  os.makedirs(args.out_dir, exist_ok=True)

  record = read_record(args.tfrecord, shape_keys, args.index)
  points, rows, cols = texel_points(record)
  points = points - (points.max(0) + points.min(0)) / 2
  points = points / np.linalg.norm(points, axis=1).max()
  print(f'{len(points)} texels, {len(sample_paths)} samples', flush=True)

  ref_colors = texel_colors(args.reference, record['tex_dim'], rows, cols)
  sample_colors = [texel_colors(p, record['tex_dim'], rows, cols)
                   for p in sample_paths]

  up = 'xyz'.index(args.up)
  elevation = np.deg2rad(args.elevation_deg)
  for v in range(args.views):
    x, y, depth = project(points, 2 * np.pi * v / args.views, elevation, up)
    vis = visibility_map(x, y, depth, args.size, args.splat)
    view_dir = os.path.join(args.out_dir, f'view_{v:02d}')
    os.makedirs(view_dir, exist_ok=True)
    Image.fromarray(shade(ref_colors, vis)).save(
        os.path.join(view_dir, 'ref.png'), compress_level=1)
    for name, colors in zip(names, sample_colors):
      Image.fromarray(shade(colors, vis)).save(
          os.path.join(view_dir, f'{name}.png'), compress_level=1)
    print(f'view {v + 1}/{args.views} done', flush=True)

  config = dict(vars(args))
  config.update(n_samples=len(sample_paths), n_texels=int(len(points)),
                git_commit=git_commit())
  with open(os.path.join(args.out_dir, 'render_config.json'), 'w') as f:
    json.dump(config, f, indent=2)


if __name__ == '__main__':
  main()