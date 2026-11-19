# coding=utf-8
"""Renders UV textures on the surface stored in a geometry record.

The geometry records store, for every valid texture pixel, the triangle that
contains it and its barycentric coordinates. This gives the 3D position of every
texel, so a texture image can be displayed on the surface as a colored point
cloud seen from several viewpoints, without any mesh with UV coordinates.

Example, from the `fldm/` directory:
  python mesh_diffusion/evaluation/render_views.py \
    --tfrecord mesh_diffusion/data/meshes/<ID>/test/geom_0.tfrecords \
    --texture mesh_diffusion/data/meshes/<ID>/valid_tex.jpg \
              mesh_diffusion/runs/ref_ddpm1000/samples/0.png \
    --out_dir results/renders
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
import tensorflow as tf


def read_record(tfrecord_path, shape_keys_path, index=0):
  """Reads one geometry example from a tfrecords file.

  Args:
    tfrecord_path: Path of the `geom_*.tfrecords` file.
    shape_keys_path: Path of the `shape_keys.json` that lists the fields, with
      their type ("int" or "float") and shape.
    index: Position of the example in the file. Only the examples before it are
      skipped, not parsed.

  Returns:
    A dict that maps each field name to a numpy array.

  Raises:
    IndexError: If the file has no example at this position.
  """
  with open(shape_keys_path) as f:
    spec = json.load(f)
  dtypes = {'int': tf.int32, 'float': tf.float32}
  dataset = tf.data.TFRecordDataset(tfrecord_path).skip(index).take(1)
  for raw in dataset:
    example = tf.train.Example.FromString(raw.numpy())
    record = {}
    for name, (kind, _) in spec.items():
      blob = example.features.feature[name].bytes_list.value[0]
      record[name] = tf.io.parse_tensor(blob, out_type=dtypes[kind]).numpy()
    return record
  raise IndexError(f'No example at index {index} in {tfrecord_path}')


def texel_points(record):
  """Computes the 3D position of every valid texture pixel.

  Args:
    record: Dict returned by `read_record`.

  Returns:
    A tuple (points, rows, cols). `points` has shape (n, 3) and `rows`, `cols`
    have shape (n,). Only the pixels flagged as valid are kept.
  """
  nodes, tris, bary = record['nodes'], record['pix_tris'], record['pix_bary']
  points = np.einsum('pk,pkc->pc', bary, nodes[tris])
  valid = record['valid_ind'] > 0.5
  return points[valid], record['I'][valid], record['J'][valid]


def texel_colors(texture_path, tex_dim, rows, cols):
  """Reads the color of the requested pixels in a texture image.

  Args:
    texture_path: Path of the texture image (png or jpg).
    tex_dim: Array (height, width) of the texture, from the record.
    rows: Row index of each pixel.
    cols: Column index of each pixel.

  Returns:
    A uint8 array of shape (n, 3).
  """
  height, width = int(tex_dim[0]), int(tex_dim[1])
  image = Image.open(texture_path).convert('RGB')
  if image.size != (width, height):
    print(f'  resizing {texture_path} from {image.size} to {(width, height)}')
    image = image.resize((width, height))
  texture = np.asarray(image)
  if rows.max() >= height or cols.max() >= width:
    raise ValueError('Pixel indices do not fit tex_dim, rows and cols may be '
                     'swapped.')
  return texture[rows, cols]


def render_view(points, colors, size, azimuth, elevation, up, splat):
  """Renders a colored point cloud from one viewpoint with a depth buffer.

  The projection is orthographic. Each point is drawn as a square of
  (2 * splat + 1) pixels. When several points fall on the same pixel the nearest
  one wins.

  Args:
    points: Float array (n, 3), already centered and scaled so that the shape
      fits in the unit ball.
    colors: Uint8 array (n, 3).
    size: Side of the square output image in pixels.
    azimuth: Rotation around the vertical axis, in radians.
    elevation: Rotation around the horizontal screen axis, in radians.
    up: Index (0, 1 or 2) of the coordinate that points up in the model.
    splat: Half-width of the square drawn for each point.

  Returns:
    A uint8 image of shape (size, size, 3) with a black background.
  """
  horiz = [i for i in range(3) if i != up]
  a, b, v = points[:, horiz[0]], points[:, horiz[1]], points[:, up]
  x = np.cos(azimuth) * a - np.sin(azimuth) * b
  depth = np.sin(azimuth) * a + np.cos(azimuth) * b
  y = np.cos(elevation) * v - np.sin(elevation) * depth
  depth = np.sin(elevation) * v + np.cos(elevation) * depth

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
  ci = np.repeat(colors, k, axis=0)
  ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
  xi, yi, di, ci = xi[ok], yi[ok], di[ok], ci[ok]

  flat = yi * size + xi
  order = np.lexsort((di, flat))  # by pixel, then nearest first
  flat, ci = flat[order], ci[order]
  first = np.concatenate([[True], flat[1:] != flat[:-1]])
  image = np.zeros((size * size, 3), np.uint8)
  image[flat[first]] = ci[first]
  return image.reshape(size, size, 3)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tfrecord', required=True)
  parser.add_argument('--shape_keys', default=None,
                      help='Defaults to shape_keys.json next to the tfrecords.')
  parser.add_argument('--index', type=int, default=0)
  parser.add_argument('--texture', nargs='+', required=True)
  parser.add_argument('--out_dir', required=True)
  parser.add_argument('--views', type=int, default=8)
  parser.add_argument('--size', type=int, default=512)
  parser.add_argument('--up', choices=['x', 'y', 'z'], default='y')
  parser.add_argument('--elevation_deg', type=float, default=15.0)
  parser.add_argument('--splat', type=int, default=1)
  args = parser.parse_args()

  shape_keys = args.shape_keys or os.path.join(
      os.path.dirname(args.tfrecord), 'shape_keys.json')
  os.makedirs(args.out_dir, exist_ok=True)

  record = read_record(args.tfrecord, shape_keys, args.index)
  points, rows, cols = texel_points(record)
  print(f'valid texels: {len(points)} / {len(record["valid_ind"])}')
  print(f'tex_dim: {record["tex_dim"]}, rows in [{rows.min()}, {rows.max()}], '
        f'cols in [{cols.min()}, {cols.max()}]')

  points = points - (points.max(0) + points.min(0)) / 2
  points = points / np.linalg.norm(points, axis=1).max()
  up = 'xyz'.index(args.up)
  elevation = np.deg2rad(args.elevation_deg)

  for path in args.texture:
    colors = texel_colors(path, record['tex_dim'], rows, cols)
    views = [
        render_view(points, colors, args.size, 2 * np.pi * i / args.views,
                    elevation, up, args.splat) for i in range(args.views)]
    per_row = min(4, args.views)
    grid_rows = [np.concatenate(views[i:i + per_row], axis=1)
                 for i in range(0, args.views, per_row)]
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(args.out_dir, f'{name}_views.png')
    Image.fromarray(np.concatenate(grid_rows, axis=0)).save(out)
    print('saved', out)


if __name__ == '__main__':
  main()
