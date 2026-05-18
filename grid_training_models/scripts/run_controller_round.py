#!/usr/bin/env python3
"""One-command Auto V5 controller round runner.

This script creates a new smoke control, materializes generated standalone model
files, submits the initial smoke jobs, then runs the local controller clock until
it reaches ``max_ticks`` or the state machine has no work left. AI code repair is
executed automatically when a repair action requires it; use
``--no-execute-repair`` only for a deterministic/config-repair-only run.
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.campaign_orchestrator import (  # noqa: E402
    DEFAULT_HARD_EXCLUDE_ARCHES,
    DEFAULT_REMOTE_ROOT,
    DEFAULT_SOFT_EXCLUDE_ARCHES,
    build_smoke_control_from_candidates,
    collect_arches_from_control_files,
    collect_sources_from_control_files,
    load_default_candidates,
    materialize_codegen_wrappers,
    write_campaign_artifacts,
)
from scripts.controller_repair_executor import attach_submit_results_to_plan  # noqa: E402
from scripts.controller_state_machine import decide_next_step  # noqa: E402

DEFAULT_HOST = "<HPC_USER>@<HPC_LOGIN>"
DEFAULT_SOCKET = "<LOCAL_HOME_PATH>"
FORBIDDEN_GENERATED_TOKENS = (
    "REGISTRY.build",
    "shared.models",
    "MODEL_REGISTRY",
    "from shared",
    "import shared",
    "shared/models",
)
RunCallable = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RoundNames:
    campaign: str
    run_prefix: str
    model_dir: str


def derive_round_names(report_dir: Path) -> RoundNames:
    """Derive campaign, run-prefix, and generated model dir from report name.

    ``reports/controller_auto10_006`` becomes the established Auto V5 naming
    scheme used by the controller artifacts.
    """
    name = report_dir.name
    token = name[len("controller_") :] if name.startswith("controller_") else name
    return RoundNames(
        campaign=f"v5_controller_{token}_smoke20",
        run_prefix=f"r_{token}",
        model_dir=f"generated_models/v5_controller_{token}",
    )


def next_report_dir(report_dir: Path) -> Path:
    """Return the next bounded controller report directory by incrementing the trailing number."""
    match = re.search(r"(\d+)$", report_dir.name)
    if not match:
        raise SystemExit(f"cannot infer next round from report dir without trailing number: {report_dir}")
    number = match.group(1)
    next_name = f"{report_dir.name[:match.start(1)]}{int(number) + 1:0{len(number)}d}"
    return report_dir.with_name(next_name)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def verify_generated_models(paths: list[Path]) -> None:
    bad: dict[str, list[str]] = {}
    for path in paths:
        py_compile.compile(str(path), doraise=True)
        text = path.read_text()
        hits = [token for token in FORBIDDEN_GENERATED_TOKENS if token in text]
        if hits:
            bad[str(path)] = hits
    if bad:
        raise SystemExit(f"generated model forbidden-token scan failed: {bad}")


def generate_initial_round(
    *,
    report_dir: Path,
    local_root: Path,
    count: int,
    remote_root: str,
    exclude_control: list[Path],
    source_campaign: str = "v5_ai_curated_001",
    include_hard_excluded_arches: bool = False,
) -> dict[str, Any]:
    names = derive_round_names(report_dir)
    candidates = load_default_candidates(local_root, model_dir=names.model_dir, source_campaign=source_campaign)
    hard_exclude_arches = set() if include_hard_excluded_arches else DEFAULT_HARD_EXCLUDE_ARCHES
    control = build_smoke_control_from_candidates(
        candidates,
        campaign=names.campaign,
        run_prefix=names.run_prefix,
        count=count,
        exclude_arches=hard_exclude_arches,
        soft_exclude_arches=DEFAULT_SOFT_EXCLUDE_ARCHES | collect_arches_from_control_files(exclude_control),
        exclude_sources=collect_sources_from_control_files(exclude_control),
        remote_root=remote_root,
    )
    outputs = write_campaign_artifacts(
        report_dir=report_dir,
        smoke_control=control,
        materialize=True,
        submit=True,
        live_crc=True,
        execute_repair=True,
    )
    generated = materialize_codegen_wrappers(local_root, control)
    verify_generated_models(generated)
    return {
        "names": names,
        "control": control,
        "outputs": outputs,
        "generated_models": generated,
    }


def _env_with_crc(*, host: str, socket: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CRC_HOST"] = host
    env["CRC_CONTROL_PATH"] = socket
    return env


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise SystemExit("expected JSON on stdout from submit command, got empty output")
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"expected JSON on stdout from submit command, got: {text[:500]}")
    return json.loads(text[start:])


def submit_initial_smoke(
    *,
    report_dir: Path,
    local_root: Path,
    control_path: Path,
    host: str,
    socket: str,
    runner: RunCallable = subprocess.run,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/launch_smoke_from_control.py",
        "--control",
        str(control_path),
        "--local-root",
        str(local_root),
        "--materialize",
        "--submit",
        "--submit-mode",
        "remote-batch",
    ]
    completed = runner(
        cmd,
        cwd=str(local_root),
        env=_env_with_crc(host=host, socket=socket),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"initial smoke submit failed rc={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    plan = _parse_json_stdout(completed.stdout)
    cluster_map = attach_submit_results_to_plan(plan)
    write_json(report_dir / "launch_plan_smoke20.submitted.json", plan)
    write_json(report_dir / "cluster_map.json", cluster_map)
    return plan


def build_clock_command(
    *,
    report_dir: Path,
    local_root: Path,
    interval_sec: int,
    max_ticks: int,
    host: str,
    socket: str,
    execute_repair: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/controller_clock.py",
        "--report-dir",
        str(report_dir),
        "--local-root",
        str(local_root),
        "--interval-sec",
        str(interval_sec),
        "--max-ticks",
        str(max_ticks),
        "--host",
        host,
        "--socket",
        socket,
    ]
    if execute_repair:
        cmd.append("--execute-repair")
    return cmd


def run_clock(
    *,
    report_dir: Path,
    local_root: Path,
    interval_sec: int,
    max_ticks: int,
    host: str,
    socket: str,
    execute_repair: bool,
    runner: RunCallable = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    cmd = build_clock_command(
        report_dir=report_dir,
        local_root=local_root,
        interval_sec=interval_sec,
        max_ticks=max_ticks,
        host=host,
        socket=socket,
        execute_repair=execute_repair,
    )
    return runner(cmd, cwd=str(local_root), env=_env_with_crc(host=host, socket=socket), text=True)


def round_is_complete(report_dir: Path) -> bool:
    if (report_dir / "final_ranking.json").exists():
        return True
    try:
        return decide_next_step(report_dir).get("decision") == "complete"
    except Exception:
        return False


def run_one_round(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "resume_existing", False) and (args.report_dir / "cluster_map.json").exists():
        summary: dict[str, Any] = {
            "report_dir": str(args.report_dir),
            "campaign": "existing",
            "generated_models": [],
            "submit": False,
            "clock": False,
            "execute_repair": bool(args.execute_repair),
            "resume_existing": True,
        }
        if not args.no_clock:
            completed = run_clock(
                report_dir=args.report_dir,
                local_root=args.local_root,
                interval_sec=args.interval_sec,
                max_ticks=args.max_ticks,
                host=args.host,
                socket=args.socket,
                execute_repair=args.execute_repair,
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            summary["clock"] = True
            summary["complete"] = round_is_complete(args.report_dir)
        return summary

    generated = generate_initial_round(
        report_dir=args.report_dir,
        local_root=args.local_root,
        count=args.count,
        remote_root=args.remote_root,
        exclude_control=args.exclude_control,
        source_campaign=args.source_campaign,
        include_hard_excluded_arches=args.include_hard_excluded_arches,
    )
    summary: dict[str, Any] = {
        "report_dir": str(args.report_dir),
        "campaign": generated["control"]["campaign"],
        "source_campaign": args.source_campaign,
        "generated_models": [str(p.relative_to(args.local_root)) for p in generated["generated_models"]],
        "submit": False,
        "clock": False,
        "execute_repair": bool(args.execute_repair),
    }

    if args.dry_run:
        return summary
    if not args.live_crc:
        raise SystemExit("live execution requires --live-crc; use --dry-run for plan/model generation only")

    plan = submit_initial_smoke(
        report_dir=args.report_dir,
        local_root=args.local_root,
        control_path=generated["outputs"]["smoke_control"],
        host=args.host,
        socket=args.socket,
    )
    summary["submit"] = True
    summary["cluster_map"] = str(args.report_dir / "cluster_map.json")
    summary["submitted_runs"] = len(plan.get("runs", []))

    if not args.no_clock:
        completed = run_clock(
            report_dir=args.report_dir,
            local_root=args.local_root,
            interval_sec=args.interval_sec,
            max_ticks=args.max_ticks,
            host=args.host,
            socket=args.socket,
            execute_repair=args.execute_repair,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        summary["clock"] = True
        summary["complete"] = round_is_complete(args.report_dir)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--count", type=int, default=10)
    p.add_argument(
        "--source-campaign",
        default="v5_ai_curated_001",
        help="curated source campaign to sample candidates from, for example v5_ai_curated_002",
    )
    p.add_argument(
        "--include-hard-excluded-arches",
        action="store_true",
        help="include known hard-excluded arches when a bounded source pool must be exhausted; expect repair/failed classifications if they still fail",
    )
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--socket", default=DEFAULT_SOCKET)
    p.add_argument("--interval-sec", type=int, default=180)
    p.add_argument("--max-ticks", type=int, default=120)
    p.add_argument("--exclude-control", action="append", type=Path, default=[])
    p.add_argument("--live-crc", action="store_true", help="Required to submit initial smoke jobs and start the controller clock.")
    p.set_defaults(execute_repair=True)
    p.add_argument("--execute-repair", action="store_true", help="Execute staged AI repair commands. This is the default for autonomous rounds.")
    p.add_argument("--no-execute-repair", action="store_false", dest="execute_repair", help="Disable staged AI repair execution; deterministic/config repair still runs.")
    p.add_argument("--no-clock", action="store_true", help="Submit smoke but do not start controller_clock.py.")
    p.add_argument("--auto-next-round", action="store_true", help="When a round reaches final ranking/complete, start the next numbered report dir until --max-rounds is reached.")
    p.add_argument("--max-rounds", type=int, default=1, help="Maximum number of rounds to start in this bounded supervisor invocation.")
    p.add_argument("--resume-existing", action="store_true", help="If cluster_map.json already exists for the first report dir, skip initial generation/submit and run the clock from existing artifacts.")
    p.add_argument("--dry-run", action="store_true", help="Generate control/model files only; no CRC submit and no clock.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.local_root = args.local_root.resolve()
    args.report_dir = args.report_dir if args.report_dir.is_absolute() else args.local_root / args.report_dir
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be >= 1")
    if args.max_rounds > 1 and not args.auto_next_round:
        raise SystemExit("--max-rounds > 1 requires --auto-next-round")

    summaries: list[dict[str, Any]] = []
    current_report_dir = args.report_dir
    for round_index in range(args.max_rounds):
        args.report_dir = current_report_dir
        summary = run_one_round(args)
        summary["round_index"] = round_index + 1
        summaries.append(summary)
        if not args.auto_next_round:
            print(json.dumps(summary, indent=2))
            return 0
        if not summary.get("complete"):
            break
        current_control = current_report_dir / "control_smoke20.json"
        if current_control.exists() and current_control not in args.exclude_control:
            args.exclude_control.append(current_control)
        current_report_dir = next_report_dir(current_report_dir)

    print(json.dumps({"auto_next_round": bool(args.auto_next_round), "rounds": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
