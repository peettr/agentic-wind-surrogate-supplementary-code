#!/usr/bin/env python3
"""Generate canonical metadata for Grid model implementations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_EXCEPTIONS = {
    "perceiver_io": "perceiver.py",
    "cno_v2": "cno.py",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to parse JSON {path}: {exc}") from exc


def source_file_for_arch(arch_name: str) -> str:
    return f"shared/models/{SOURCE_EXCEPTIONS.get(arch_name, arch_name + '.py')}"


def build_metadata(architectures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arch in architectures:
        if not arch.get("include_in_v5", True):
            continue
        name = arch["arch_name"]
        rows.append(
            {
                "arch_name": name,
                "source_file": source_file_for_arch(name),
                "registry_key": name,
                "group": arch.get("group", "unknown"),
                "v3_status": arch.get("v3_status", "unknown"),
                "code_source": "reused_v3",
                "default_arch_kwargs": {},
                "input_channels": 1,
                "output_channels": 1,
                "main_benchmark_eligible": True,
                "notes": arch.get("notes", ""),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architectures", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = load_json(args.architectures)
    if not isinstance(data, list):
        raise SystemExit("ERROR: architectures root must be a list")
    metadata = build_metadata(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(metadata)} model metadata entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



