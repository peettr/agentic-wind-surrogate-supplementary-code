"""preprocess_sdf.py — Precompute SDF + normal channels for UrbanTALES.

Reads raw topo files, computes SDF and boundary normal angle,
saves as 3-channel numpy arrays (height, sdf, normal_angle).

Output: data/lu_sdf_640/<case_name>.npy  shape (3, H, W)

Usage:
    python -m shared.preprocess_sdf --data-dir /path/to/raw --out-dir data/lu_sdf_640
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure shared is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from shared.sdf import augment_input_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/lu_sdf_640"))
    parser.add_argument("--no-flipud", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted([d for d in args.data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases")

    done = 0
    for i, case_dir in enumerate(cases):
        topo_files = list(case_dir.glob("*_topo"))
        if not topo_files:
            continue
        topo = np.loadtxt(topo_files[0], dtype=np.float32)
        if not args.no_flipud:
            topo = np.flipud(topo)

        channels = augment_input_channels(topo)
        np.save(args.out_dir / f"{case_dir.name}.npy", channels)
        done += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)} ({done} ok)")

    print(f"Done: {done}/{len(cases)} saved to {args.out_dir}")


if __name__ == "__main__":
    main()
