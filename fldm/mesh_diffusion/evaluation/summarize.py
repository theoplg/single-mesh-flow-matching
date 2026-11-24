# coding=utf-8
"""Collects the metrics of several sampling runs into one table.

Example, from the `fldm/` directory:
  python mesh_diffusion/evaluation/summarize.py \
    --runs_dir mesh_diffusion/runs --out results/eval_summary.csv
"""
import argparse
import csv
import glob
import json
import os
import re


def parse_wall_seconds(meta_path):
  """Reads the `wall_seconds:` line of a run's meta.txt, None if absent."""
  if not os.path.exists(meta_path):
    return None
  with open(meta_path) as f:
    for line in f:
      match = re.match(r'wall_seconds:\s*(\d+)', line)
      if match:
        return int(match.group(1))
  return None


def parse_steps(run):
  """Guesses the number of network evaluations from the run name."""
  match = re.search(r'_n(\d+)(?:_|$)', run) or re.search(r'ddpm(\d+)', run)
  return int(match.group(1)) if match else None


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runs_dir', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--metrics_file', default='metrics.json',
                      help='Metrics file to read in each run directory.')
  args = parser.parse_args()

  rows = []
  for path in sorted(glob.glob(os.path.join(args.runs_dir, '*', args.metrics_file))):
    run_dir = os.path.dirname(path)
    run = os.path.basename(run_dir)
    with open(path) as f:
      m = json.load(f)
    rows.append({
        'run': run,
        'steps': parse_steps(run),
        'n_samples': m['n_samples'],
        'sifid_mean': m['sifid_mean'],
        'sifid_std_over_samples': m['sifid_std_over_samples'],
        'lpips_mean': m['lpips_mean'],
        'wall_seconds': parse_wall_seconds(os.path.join(run_dir, 'meta.txt')),
    })
  if not rows:
    raise SystemExit(f'No metrics.json found in {args.runs_dir}')
  rows.sort(key=lambda r: (r['steps'] is None, -(r['steps'] or 0), r['run']))

  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  print('| run | steps | n | SIFID | std | LPIPS | wall (s) |')
  print('|---|---|---|---|---|---|---|')
  for r in rows:
    print(f"| {r['run']} | {r['steps']} | {r['n_samples']} | "
          f"{r['sifid_mean']:.4f} | {r['sifid_std_over_samples']:.4f} | "
          f"{r['lpips_mean']:.4f} | {r['wall_seconds']} |")
  print('saved', args.out)


if __name__ == '__main__':
  main()