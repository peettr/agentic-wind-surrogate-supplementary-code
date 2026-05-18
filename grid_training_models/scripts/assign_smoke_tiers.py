#!/usr/bin/env python3
"""Assign Auto V5 codegen smoke runs to Condor GPU tiers.

Input is a dynamic-precheck JSON containing rows with at least:
  arch_name, run_id, params, model_file

Policy follows the human researcher's preference to use available GPUs broadly while keeping
resource safeguards:
  * params > 150M are not submitted;
  * FFT/FNO/spectral models go to high-memory H100/A100/L40S even if params are low;
  * KAN, multiscale, and NAF-style models also go to high-memory tier because
    batch004 showed RTX6000 OOM / high activation memory despite modest params;
  * >=120M params go to high-memory tier;
  * otherwise use A40/RTX6000 tier by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PARAM_LIMIT = 150_000_000
HIGH_PARAM_THRESHOLD = 120_000_000
HIGH_ACTIVATION_TOKENS = (
    "fno",
    "fourier",
    "spectral",
    "afno",
    "ufno",
    "ffno",
    "kan",
    "multiscale",
    "naf",
)


def is_high_activation_like(row: dict) -> bool:
    haystack = " ".join(str(row.get(k, "")).lower() for k in ("arch_name", "run_id", "model_file"))
    return any(tok in haystack for tok in HIGH_ACTIVATION_TOKENS)


def assign_one(row: dict) -> dict:
    params = int(row.get("params", -1))
    out = dict(row)
    out["param_limit"] = PARAM_LIMIT
    if params < 0:
        out["submit_tier"] = "DO_NOT_SUBMIT"
        out["reason"] = "missing_params"
    elif params > PARAM_LIMIT:
        out["submit_tier"] = "DO_NOT_SUBMIT"
        out["reason"] = "param_limit_exceeded"
    elif is_high_activation_like(row):
        out["submit_tier"] = "h100_a100_l40s_12gb"
        out["reason"] = "high_activation_memory_pattern"
    elif params >= HIGH_PARAM_THRESHOLD:
        out["submit_tier"] = "h100_a100_l40s_12gb"
        out["reason"] = "high_param_count"
    else:
        out["submit_tier"] = "a40_rtx6k_16gb"
        out["reason"] = "broad_gpu_default"
    return out


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if "models" in data:
        return data["models"]
    if "entries" in data:
        return data["entries"]
    raise ValueError("precheck JSON must be a list or contain 'models'/'entries'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--precheck", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = load_rows(args.precheck)
    assignments = [assign_one(row) for row in rows]
    payload = {
        "policy": {
            "param_limit": PARAM_LIMIT,
            "high_param_threshold": HIGH_PARAM_THRESHOLD,
            "high_activation_tokens": list(HIGH_ACTIVATION_TOKENS),
            "default_submit_tier": "a40_rtx6k_16gb",
        },
        "summary": {
            "total": len(assignments),
            "do_not_submit": sum(1 for r in assignments if r["submit_tier"] == "DO_NOT_SUBMIT"),
            "h100_a100_l40s_12gb": sum(1 for r in assignments if r["submit_tier"] == "h100_a100_l40s_12gb"),
            "a40_rtx6k_16gb": sum(1 for r in assignments if r["submit_tier"] == "a40_rtx6k_16gb"),
        },
        "assignments": assignments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
