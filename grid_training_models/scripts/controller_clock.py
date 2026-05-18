#!/usr/bin/env python3
"""Local Grid controller clock.

Runs controller-owned ticks from WSL. Each tick inspects the generic state
machine and starts the appropriate bounded program, including smoke repair,
without relying on Hermes cron prompts to decide whether to act. Repair-time AI
execution is enabled by default, while the repair executor still constrains
changes to generated standalone model files unless shared-code edits are
separately authorized.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.controller_state_machine import decide_next_step, write_json

DEFAULT_HOST = "<HPC_USER>@<HPC_LOGIN>"
DEFAULT_SOCKET = "<LOCAL_HOME_PATH>"

Runner = Callable[[str], int]
Preflight = Callable[[], bool]


def shell_runner(cmd: str) -> int:
    return subprocess.run(cmd, shell=True).returncode


def make_preflight(host: str, socket: str) -> Preflight:
    def _preflight() -> bool:
        cmd = (
            "/usr/bin/timeout -k 3s 30s ssh -o BatchMode=yes "
            f"-o ControlPath={socket} -o ConnectTimeout=20 {host!s} 'echo OK; hostname; date'"
        )
        return shell_runner(cmd) == 0
    return _preflight


def _base_env(host: str, socket: str) -> str:
    return f"CRC_HOST={host} CRC_CONTROL_PATH={socket}"


def _run_smoke_repair(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str, execute_repair: bool = True) -> None:
    log = report_dir / "smoke_repair_live_submit.log"
    execute_flag = "--execute-repair " if execute_repair else ""
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step smoke-repair "
        f"--report-dir {report_dir} --local-root . "
        f"--stage-repair {execute_flag}--materialize-repair --submit-repair --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"smoke repair submit failed with exit code {rc}; see {log}")


def _monitor_repair(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'repair_control.smoke20.json'} "
        f"--launch-plan {report_dir / 'repair_submit_plan.smoke20.json'} "
        f"--cluster-map {report_dir / 'repair_cluster_map.smoke20.json'} "
        f"--state {report_dir / 'controller_state.smoke20.repair1.json'} "
        f"--events {report_dir / 'controller_events.smoke20.repair1.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.smoke20.repair1.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"repair monitor failed with exit code {rc}")


def _monitor_repair_repair(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'repair_repair_control.smoke20.repair1.json'} "
        f"--launch-plan {report_dir / 'repair_repair_submit_plan.smoke20.repair1.json'} "
        f"--cluster-map {report_dir / 'repair_repair_cluster_map.smoke20.repair1.json'} "
        f"--state {report_dir / 'controller_state.smoke20.repair1.repair1.json'} "
        f"--events {report_dir / 'controller_events.smoke20.repair1.repair1.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.smoke20.repair1.repair1.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"second repair monitor failed with exit code {rc}")


def _state_has_pending_action(state_path: Path, cluster_map_path: Path, action_name: str) -> bool:
    if not state_path.exists() or cluster_map_path.exists():
        return False
    data = json.loads(state_path.read_text())
    runs = data.get("runs") if isinstance(data, dict) else {}
    if not isinstance(runs, dict):
        return False
    action_name = action_name.upper()
    for row in runs.values():
        if not isinstance(row, dict):
            continue
        plan = row.get("plan") or {}
        action = str(plan.get("action") or "").upper() if isinstance(plan, dict) else ""
        state_key = str(row.get("state_key") or "")
        if action == action_name or state_key.endswith(f":{action_name}"):
            return True
    return False


def _state_has_pending_retry(state_path: Path, cluster_map_path: Path) -> bool:
    return _state_has_pending_action(state_path, cluster_map_path, "RETRY")


def _state_has_pending_repair(state_path: Path, cluster_map_path: Path) -> bool:
    return _state_has_pending_action(state_path, cluster_map_path, "REPAIR")


def _pending_attempt_run_ids(state_path: Path, action_name: str) -> set[str]:
    if not state_path.exists():
        return set()
    data = json.loads(state_path.read_text())
    runs = data.get("runs") if isinstance(data, dict) else {}
    if not isinstance(runs, dict):
        return set()
    action_name = action_name.upper()
    pending: set[str] = set()
    for source_run_id, row in runs.items():
        if not isinstance(row, dict):
            continue
        plan = row.get("plan") or {}
        if not isinstance(plan, dict):
            plan = {}
        action = str(plan.get("action") or "").upper()
        state_key = str(row.get("state_key") or "")
        if action != action_name and not state_key.endswith(f":{action_name}"):
            continue
        new_run_id = plan.get("new_run_id")
        if isinstance(new_run_id, str) and new_run_id:
            pending.add(new_run_id)
        elif isinstance(row.get("current_run_id"), str):
            pending.add(str(row["current_run_id"]))
        else:
            pending.add(str(source_run_id))
    return pending


def _cluster_map_run_ids(cluster_map_path: Path) -> set[str]:
    if not cluster_map_path.exists():
        return set()
    data = json.loads(cluster_map_path.read_text())
    if not isinstance(data, dict):
        return set()
    return {str(run_id) for run_id in data.keys()}


def _state_has_unsubmitted_action(state_path: Path, cluster_map_path: Path, action_name: str) -> bool:
    return bool(_pending_attempt_run_ids(state_path, action_name) - _cluster_map_run_ids(cluster_map_path))


def _run_smoke_repair_retry(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    log = report_dir / "smoke_repair_retry_live_submit.log"
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step smoke-repair-retry "
        f"--report-dir {report_dir} --local-root . "
        "--materialize-retry --submit-retry --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"smoke repair retry submit failed with exit code {rc}; see {log}")


def _run_smoke_repair_repair(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str, execute_repair: bool = True) -> None:
    log = report_dir / "smoke_repair_repair_live_submit.log"
    execute_flag = "--execute-repair " if execute_repair else ""
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step smoke-repair-repair "
        f"--report-dir {report_dir} --local-root . "
        f"--stage-repair {execute_flag}--materialize-repair --submit-repair --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"smoke repair repair submit failed with exit code {rc}; see {log}")


def _run_smoke_retry(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    log = report_dir / "smoke_retry_live_submit.log"
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step smoke-retry "
        f"--report-dir {report_dir} --local-root . "
        "--materialize-retry --submit-retry --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"smoke retry submit failed with exit code {rc}; see {log}")


def _monitor_smoke_retry(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'retry_control.smoke20.json'} "
        f"--launch-plan {report_dir / 'retry_submit_plan.smoke20.json'} "
        f"--cluster-map {report_dir / 'retry_cluster_map.smoke20.json'} "
        f"--state {report_dir / 'controller_state.smoke20.retry1.json'} "
        f"--events {report_dir / 'controller_events.smoke20.retry1.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.smoke20.retry1.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"smoke retry monitor failed with exit code {rc}")


def _monitor_smoke(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    if not (report_dir / "cluster_map.json").exists():
        return
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'control_smoke20.json'} "
        f"--launch-plan {report_dir / 'launch_plan_smoke20.submitted.json'} "
        f"--cluster-map {report_dir / 'cluster_map.json'} "
        f"--state {report_dir / 'controller_state.smoke20.json'} "
        f"--events {report_dir / 'controller_events.smoke20.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.smoke20.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    runner(cmd)


def _benchmark_cluster_map(report_dir: Path) -> Path:
    full = report_dir / "benchmark_cluster_map.full.json"
    if full.exists():
        return full
    return report_dir / "benchmark_cluster_map.json"


def _monitor_benchmark(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    cluster_map = _benchmark_cluster_map(report_dir)
    if not cluster_map.exists():
        return
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'control_benchmark200.json'} "
        f"--launch-plan {report_dir / 'launch_plan_benchmark200.submitted.json'} "
        f"--cluster-map {cluster_map} "
        f"--state {report_dir / 'controller_state.benchmark200.json'} "
        f"--events {report_dir / 'controller_events.benchmark200.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.benchmark200.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"benchmark monitor failed with exit code {rc}")


def _monitor_benchmark_retry(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    cmd = (
        f"cd {local_root} && python scripts/controller_monitor.py "
        f"--control {report_dir / 'retry_control.benchmark200.json'} "
        f"--launch-plan {report_dir / 'retry_submit_plan.benchmark200.json'} "
        f"--cluster-map {report_dir / 'retry_cluster_map.benchmark200.json'} "
        f"--state {report_dir / 'controller_state.benchmark200.retry1.json'} "
        f"--events {report_dir / 'controller_events.benchmark200.retry1.jsonl'} "
        f"--notify-file {report_dir / 'controller_notifications.benchmark200.retry1.md'} "
        f"--once --monitor-only --live-crc --ssh-host {host} --ssh-control-path {socket}"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"benchmark retry monitor failed with exit code {rc}")


def _run_benchmark_retry(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    log = report_dir / "benchmark_retry_live_submit.log"
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step benchmark-retry "
        f"--report-dir {report_dir} --local-root . "
        "--materialize-retry --submit-retry --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"benchmark retry submit failed with exit code {rc}; see {log}")


def _run_smoke_to_benchmark(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    promote_log = report_dir / "smoke_to_benchmark_live.log"
    promote_cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step smoke-to-benchmark "
        f"--report-dir {report_dir} --local-root . "
        "--materialize-benchmark --live-crc "
        f"> {promote_log} 2>&1"
    )
    rc = runner(promote_cmd)
    if rc != 0:
        raise SystemExit(f"smoke-to-benchmark materialize failed with exit code {rc}; see {promote_log}")

    submit_log = report_dir / "benchmark_live_submit.log"
    submit_cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step submit-benchmark "
        f"--report-dir {report_dir} --local-root . "
        "--submit-benchmark --live-crc "
        f"> {submit_log} 2>&1"
    )
    rc = runner(submit_cmd)
    if rc != 0:
        raise SystemExit(f"benchmark submit failed with exit code {rc}; see {submit_log}")


def _run_final_ranking(*, report_dir: Path, local_root: Path, runner: Runner, host: str, socket: str) -> None:
    log = report_dir / "final_ranking_live.log"
    cmd = (
        f"cd {local_root} && {_base_env(host, socket)} "
        "python scripts/controller_driver.py --step auto-advance "
        f"--report-dir {report_dir} --local-root . --live-crc "
        f"> {log} 2>&1"
    )
    rc = runner(cmd)
    if rc != 0:
        raise SystemExit(f"final ranking failed with exit code {rc}; see {log}")


def controller_tick(
    *,
    report_dir: Path,
    local_root: Path,
    runner: Runner = shell_runner,
    preflight: Preflight | None = None,
    host: str = DEFAULT_HOST,
    socket: str = DEFAULT_SOCKET,
    execute_repair: bool = True,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    local_root = Path(local_root)
    if preflight is None:
        preflight = make_preflight(host, socket)
    if not preflight():
        result = {"decision": "socket-unavailable", "advanced": False}
        write_json(report_dir / "controller_clock_last_tick.json", result)
        return result

    decision = decide_next_step(report_dir)
    result: dict[str, Any] = {**decision, "advanced": False}

    # Keep original smoke state fresh, but never let monitor-only replace action decisions.
    if (report_dir / "cluster_map.json").exists():
        _monitor_smoke(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
        decision = decide_next_step(report_dir)
        result.update(decision)

    if decision.get("decision") == "smoke-retry":
        retry_map = report_dir / "retry_cluster_map.smoke20.json"
        smoke_state = report_dir / "controller_state.smoke20.json"
        if retry_map.exists() and _state_has_unsubmitted_action(smoke_state, retry_map, "RETRY"):
            _run_smoke_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"advanced": True, "action": "submitted-missing-smoke-retry"})
        elif retry_map.exists():
            _monitor_smoke_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"decision": "smoke-retry-monitor", "advanced": True})
        else:
            _run_smoke_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"advanced": True, "action": "submitted-smoke-retry"})

    if decision.get("decision") == "smoke-repair":
        if (report_dir / "repair_cluster_map.smoke20.json").exists():
            repair_retry_map = report_dir / "repair_retry_cluster_map.smoke20.repair1.json"
            repair_repair_map = report_dir / "repair_repair_cluster_map.smoke20.repair1.json"
            repair1_state = report_dir / "controller_state.smoke20.repair1.json"
            if repair_repair_map.exists():
                _monitor_repair_repair(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
                result.update({"decision": "smoke-repair-repair-monitor", "advanced": True})
            else:
                _monitor_repair(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
                if _state_has_pending_retry(repair1_state, repair_retry_map):
                    _run_smoke_repair_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
                    result.update({"decision": "smoke-repair-retry", "advanced": True, "action": "submitted-smoke-repair-retry"})
                elif _state_has_pending_repair(repair1_state, repair_repair_map):
                    _run_smoke_repair_repair(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket, execute_repair=execute_repair)
                    result.update({"decision": "smoke-repair-repair", "advanced": True, "action": "submitted-smoke-repair-repair"})
                else:
                    result.update({"decision": "smoke-repair-monitor", "advanced": True})
        else:
            _run_smoke_repair(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket, execute_repair=execute_repair)
            result.update({"advanced": True, "action": "submitted-smoke-repair"})

    if decision.get("decision") == "smoke-to-benchmark":
        if (report_dir / "benchmark_cluster_map.json").exists() or (report_dir / "benchmark_cluster_map.full.json").exists():
            result.update({"decision": "smoke-to-benchmark", "advanced": False, "action": "benchmark-already-submitted"})
        else:
            _run_smoke_to_benchmark(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"decision": "smoke-to-benchmark", "advanced": True, "action": "submitted-benchmark"})

    if decision.get("decision") == "benchmark-monitor":
        _monitor_benchmark(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
        result.update({"decision": "benchmark-monitor", "advanced": True})

    if decision.get("decision") == "benchmark-retry":
        retry_map = report_dir / "retry_cluster_map.benchmark200.json"
        benchmark_state = report_dir / "controller_state.benchmark200.json"
        if retry_map.exists() and _state_has_unsubmitted_action(benchmark_state, retry_map, "RETRY"):
            _run_benchmark_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"advanced": True, "action": "submitted-missing-benchmark-retry"})
        elif retry_map.exists():
            _monitor_benchmark_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"decision": "benchmark-retry-monitor", "advanced": True})
        else:
            _run_benchmark_retry(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"advanced": True, "action": "submitted-benchmark-retry"})

    if decision.get("decision") == "final-ranking":
        if (report_dir / "final_ranking.json").exists():
            result.update({"decision": "complete", "advanced": False, "action": "final-ranking-already-exists"})
        else:
            _run_final_ranking(report_dir=report_dir, local_root=local_root, runner=runner, host=host, socket=socket)
            result.update({"decision": "final-ranking", "advanced": True, "action": "wrote-final-ranking"})

    write_json(report_dir / "controller_clock_last_tick.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--local-root", type=Path, default=Path("."))
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--max-ticks", type=int, default=1)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--socket", default=DEFAULT_SOCKET)
    p.set_defaults(execute_repair=True)
    p.add_argument("--execute-repair", action="store_true", help="Execute staged AI repair commands. This is the default for autonomous rounds.")
    p.add_argument("--no-execute-repair", action="store_false", dest="execute_repair", help="Disable staged AI repair execution; deterministic/config repair still runs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for i in range(args.max_ticks):
        result = controller_tick(report_dir=args.report_dir, local_root=args.local_root, host=args.host, socket=args.socket, execute_repair=args.execute_repair)
        print(json.dumps(result, indent=2))
        if result.get("decision") == "complete":
            break
        if i + 1 < args.max_ticks:
            time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



