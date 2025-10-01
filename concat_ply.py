#!/usr/bin/env python3
import argparse
import glob
import re
import sys
from pathlib import Path

import open3d as o3d
import numpy as np

def natsort_key(s: str):
    # Natural sort: "pc_2.ply" before "pc_10.ply"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def concat_pointclouds(files):
    if not files:
        raise ValueError("No input .ply files found.")
    print(files)
    pcs = []
    for f in files:
        pcd = o3d.io.read_point_cloud(f)
        if pcd.is_empty():
            print(f"Warning: {f} is empty; skipping.", file=sys.stderr)
            continue
        pcs.append(pcd)

    if not pcs:
        raise ValueError("All input .ply files were empty.")

    # Determine which attributes are consistently present
    all_have_colors  = all(p.has_colors()  for p in pcs)
    all_have_normals = all(p.has_normals() for p in pcs)

    # Concatenate points (and attrs if available)
    pts = np.vstack([np.asarray(p.points) for p in pcs])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)

    if all_have_colors:
        cols = np.vstack([np.asarray(p.colors) for p in pcs])
        out.colors = o3d.utility.Vector3dVector(cols)
    if all_have_normals:
        nrm = np.vstack([np.asarray(p.normals) for p in pcs])
        out.normals = o3d.utility.Vector3dVector(nrm)

    return out, all_have_colors, all_have_normals

def main():
    ap = argparse.ArgumentParser(description="Concatenate up to 36 .ply point clouds into out.ply")
    ap.add_argument("input", nargs="?", default=".", help="Input directory or glob (default: current dir)")
    ap.add_argument("-p", "--pattern", default="*.ply", help="Glob pattern inside directory (default: *.ply)")
    ap.add_argument("-n", "--num", type=int, default=36, help="Max number of files to combine (default: 36)")
    ap.add_argument("-o", "--output", default="out.ply", help="Output filename (default: out.ply)")
    ap.add_argument("--ascii", action="store_true", help="Write ASCII PLY (default: binary)")
    args = ap.parse_args()

    # Resolve file list
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(glob.glob(str(input_path / args.pattern)), key=natsort_key)
    else:
        # Treat args.input as a glob itself
        files = sorted(glob.glob(args.input), key=natsort_key)

    files = files[: args.num]
    if not files:
        print("No files matched. Check your path/pattern.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files. Concatenating…")
    for i, f in enumerate(files, 1):
        print(f"[{i:02d}] {f}")

    merged, has_cols, has_norms = concat_pointclouds(files)

    success = o3d.io.write_point_cloud(args.output, merged, write_ascii=args.ascii)
    if not success:
        print("Failed to write output point cloud.", file=sys.stderr)
        sys.exit(2)

    print(f"\nWrote {args.output} with {len(merged.points)} points."
          f"{' Colors kept.' if has_cols else ' No consistent colors.'}"
          f"{' Normals kept.' if has_norms else ' No consistent normals.'}")

if __name__ == "__main__":
    main()

