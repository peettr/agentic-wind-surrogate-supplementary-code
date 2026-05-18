#!/usr/bin/env python3
"""Condor job submitter with preflight checks and post-submit monitoring.

Usage:
    python3 condor_submit_safe.py <submit_file> <campaign_dir>

Steps:
    1. Preflight: validate configs, clean stale files, fix permissions
    2. Submit via condor_submit
    3. Post-submit: poll every 60s, auto-diagnose held jobs, release when fixed
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a shell command via SSH on CRC login node."""
    return subprocess.run(
        ["wsl", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _crc_ssh(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run command on CRC via ControlMaster socket. Returns (rc, stdout, stderr)."""
    control_path = os.environ.get("CRC_CONTROL_PATH", "<SSH_CONTROL_PATH>")
    host = os.environ.get("CRC_HOST", "<HPC_USER>@<HPC_LOGIN>")
    ssh = f"ssh -o ConnectTimeout=120 -o ServerAliveInterval=10 -o ControlPath={control_path} {host}"
    full = f'{ssh} {cmd}'
    r = _run(full, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _crc_scp(local: str, remote: str) -> int:
    """SCP local file to CRC."""
    control_path = os.environ.get("CRC_CONTROL_PATH", "<SSH_CONTROL_PATH>")
    host = os.environ.get("CRC_HOST", "<HPC_USER>@<HPC_LOGIN>")
    scp = f"scp -o ConnectTimeout=120 -o ControlPath={control_path} {local} {host}:{remote}"
    r = _run(scp, timeout=30)
    return r.returncode


def parse_run_names(submit_path: str) -> list[str]:
    """Parse run names from a Condor submit file."""
    names = []
    with open(submit_path) as f:
        for line in f:
            m = re.match(r"^Name\s*=\s*(.+)", line.strip())
            if m:
                names.append(m.group(1).strip())
    return names


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

class PreflightCheck:
    def __init__(self, campaign_dir: str, run_names: list[str]):
        self.campaign = campaign_dir
        self.runs = run_names
        self.issues: list[str] = []
        self.fixes: list[str] = []

    def _crc(self, cmd: str) -> tuple[int, str]:
        rc, out, err = _crc_ssh(f"'{cmd}'")
        return rc, out

    def check_configs_exist(self) -> bool:
        missing = []
        for name in self.runs:
            cfg = f"{self.campaign}/{name}/train_config.json"
            rc, _ = self._crc(f"test -f {cfg}")
            if rc != 0:
                missing.append(name)
        if missing:
            self.issues.append(f"Missing train_config.json: {missing}")
            return False
        print(f"  âœ… All {len(self.runs)} train_config.json present")
        return True

    def check_config_fields(self) -> bool:
        required = ["experiment_id", "seed", "epochs", "lr", "batch_size",
                     "arch_name", "loss_name", "data_dir", "results_dir",
                     "split_manifest_path"]
        bad = []
        for name in self.runs:
            cfg_path = f"{self.campaign}/{name}/train_config.json"
            rc, out = self._crc(f"python3 -c \"import json; c=json.load(open('{cfg_path}')); "
                                f"missing=[f for f in {required} if f not in c]; "
                                f"print(','.join(missing)) if missing else print('ok')\"")
            if rc != 0 or out != "ok":
                bad.append(f"{name}({out})")
        if bad:
            self.issues.append(f"Bad configs: {bad}")
            return False
        print(f"  âœ… All configs valid")
        return True

    def check_compute_r2(self) -> bool:
        no_r2 = []
        for name in self.runs:
            cfg_path = f"{self.campaign}/{name}/train_config.json"
            rc, out = self._crc(f"python3 -c \"import json; c=json.load(open('{cfg_path}')); "
                                f"print(c.get('compute_r2', False))\"")
            if out.strip() != "True":
                no_r2.append(name)
        if no_r2:
            self.fixes.append(f"âš ï¸  compute_r2 not set in: {no_r2}")
            return False
        print(f"  âœ… compute_r2=True in all configs")
        return True

    def clean_stale_files(self) -> None:
        """Remove stale output/error/sentinel/model files from previous runs."""
        stale_patterns = ["*.out", "*.err", "STARTED", "FINISHED", "FAILED",
                          "HEARTBEAT.json", "model_best.pt", "checkpoint.pt",
                          "train.log"]
        cleaned = 0
        for name in self.runs:
            for pat in stale_patterns:
                f = f"{self.campaign}/{name}/{pat}" if not pat.startswith("*") else None
                if pat.startswith("*"):
                    # .out and .err are in campaign_dir, not run dir
                    ext = pat.replace("*", name)
                    f = f"{self.campaign}/{ext}"
                if f:
                    rc, _ = self._crc(f"rm -f {f}")
                    cleaned += 1
            # Also clean campaign-level .out/.err
            for ext in [".out", ".err"]:
                f = f"{self.campaign}/{name}{ext}"
                self._crc(f"rm -f {f}")
        # campaign-level log
        self._crc(f"rm -f {self.campaign}/condor.log")
        print(f"  âœ… Cleaned stale files for {len(self.runs)} runs")

    def check_permissions(self) -> bool:
        """Ensure run dirs are writable."""
        bad = []
        for name in self.runs:
            d = f"{self.campaign}/{name}"
            rc, _ = self._crc(f"chmod 755 {d} 2>/dev/null")
        print(f"  âœ… Run dir permissions OK")
        return True

    def check_data_files(self) -> bool:
        """Check all_data.pt and split_manifest.json exist."""
        cfg_path = f"{self.campaign}/{self.runs[0]}/train_config.json"
        rc, out = self._crc(f"python3 -c \"import json; c=json.load(open('{cfg_path}')); print(c['data_dir'])\"")
        if rc != 0:
            self.issues.append("Cannot read data_dir from config")
            return False
        data_dir = out.strip()
        for f in ["all_data.pt", "split_manifest.json"]:
            rc, _ = self._crc(f"test -f {data_dir}/{f}")
            if rc != 0:
                self.issues.append(f"Missing: {data_dir}/{f}")
                return False
        print(f"  âœ… Data files present in {data_dir}")
        return True

    def check_wrapper(self, submit_path: str) -> bool:
        """Check condor_wrapper.sh exists."""
        with open(submit_path) as f:
            for line in f:
                m = re.match(r"executable\s*=\s*(.+)", line.strip())
                if m:
                    wrapper = m.group(1).strip()
                    rc, _ = self._crc(f"test -f {wrapper}")
                    if rc != 0:
                        self.issues.append(f"Missing wrapper: {wrapper}")
                        return False
                    print(f"  âœ… Wrapper exists: {wrapper}")
                    return True
        self.issues.append("No executable line in submit file")
        return False

    def run_all(self, submit_path: str) -> bool:
        print("\n=== Preflight Check ===")
        self.clean_stale_files()
        self.check_permissions()
        self.check_configs_exist()
        self.check_config_fields()
        self.check_compute_r2()
        self.check_data_files()
        self.check_wrapper(submit_path)

        print(f"\n  Fixes: {len(self.fixes)}")
        for f in self.fixes:
            print(f"    {f}")

        if self.issues:
            print(f"\nâŒ {len(self.issues)} issues found:")
            for i in self.issues:
                print(f"    {i}")
            return False

        print("\nâœ… All checks passed â€” safe to submit")
        return True


# ---------------------------------------------------------------------------
# Post-submit monitor
# ---------------------------------------------------------------------------

class PostSubmitMonitor:
    def __init__(self, cluster_id: int, campaign_dir: str, run_names: list[str],
                 poll_interval: int = 60, max_polls: int = 5):
        self.cluster = cluster_id
        self.campaign = campaign_dir
        self.runs = set(run_names)
        self.poll_interval = poll_interval
        self.max_polls = max_polls

    def _crc(self, cmd: str) -> tuple[int, str]:
        rc, out, _ = _crc_ssh(f"'{cmd}'")
        return rc, out

    def get_status(self) -> dict[str, int]:
        """Returns {status_code: count} for the cluster."""
        rc, out = self._crc(f"condor_q {self.cluster} -af JobStatus 2>/dev/null")
        if rc != 0:
            return {}
        counts: dict[int, int] = {}
        for line in out.strip().split("\n"):
            line = line.strip()
            if line.isdigit():
                s = int(line)
                counts[s] = counts.get(s, 0) + 1
        return counts

    def get_held_reasons(self) -> list[dict]:
        """Get details of held jobs."""
        rc, out = self._crc(
            f'condor_q {self.cluster} -af Name JobStatus HoldReason 2>/dev/null'
        )
        if rc != 0:
            return []
        held = []
        lines = out.strip().split("\n")
        for i in range(0, len(lines) - 1, 3):
            name = lines[i].strip() if i < len(lines) else ""
            status = lines[i+1].strip() if i+1 < len(lines) else ""
            reason = lines[i+2].strip() if i+2 < len(lines) else ""
            if status == "5":
                held.append({"name": name, "reason": reason})
        return held

    def fix_and_release(self, held_jobs: list[dict]) -> int:
        """Try to fix held jobs and release them."""
        fixed = 0
        for job in held_jobs:
            reason = job["reason"]
            name = job["name"]
            print(f"  ðŸ”§ Fixing held job {name}: {reason[:80]}...")

            # Permission denied on output/error file
            if "Permission denied" in reason:
                match = re.search(r"Failed to open '([^']+)'", reason)
                if match:
                    bad_file = match.group(1)
                    self._crc(f"rm -f {bad_file}")
                    print(f"    Removed: {bad_file}")
                    fixed += 1

            # Other file issues
            if "No such file" in reason or "error" in reason.lower():
                self._crc(f"rm -f {self.campaign}/{name}.out {self.campaign}/{name}.err")
                fixed += 1

        if fixed > 0:
            print(f"  Releasing cluster {self.cluster}...")
            self._crc(f"condor_release {self.cluster}")
        return fixed

    def run(self) -> bool:
        """Monitor jobs for max_polls iterations. Returns True if all running/completed."""
        print(f"\n=== Post-Submit Monitor (cluster {self.cluster}) ===")
        print(f"Polling every {self.poll_interval}s for {self.max_polls} checks\n")

        STATUS_MEANING = {
            1: "Idle", 2: "Running", 3: "Removed", 4: "Completed",
            5: "Held", 6: "Transferring"
        }

        for poll in range(1, self.max_polls + 1):
            counts = self.get_status()
            if not counts:
                print(f"  Poll {poll}/{self.max_polls}: cannot reach CRC, waiting...")
                time.sleep(self.poll_interval)
                continue

            total = sum(counts.values())
            held = counts.get(5, 0)
            running = counts.get(2, 0)
            idle = counts.get(1, 0)
            done = counts.get(4, 0)

            summary = ", ".join(
                f"{STATUS_MEANING.get(k, f'?{k}')}: {v}"
                for k, v in sorted(counts.items())
            )
            print(f"  Poll {poll}/{self.max_polls}: {summary}")

            if held > 0:
                print(f"  âš ï¸  {held} job(s) held â€” diagnosing...")
                held_jobs = self.get_held_reasons()
                for j in held_jobs:
                    print(f"    {j['name']}: {j['reason'][:100]}")
                n_fixed = self.fix_and_release(held_jobs)
                if n_fixed > 0:
                    print(f"  âœ… Fixed {n_fixed}, released")

            if idle == 0 and held == 0 and running > 0:
                print(f"  âœ… All {total} jobs running!")
                return True

            if done == total:
                print(f"  âœ… All {total} jobs completed!")
                return True

            time.sleep(self.poll_interval)

        print(f"\n  âš ï¸  Monitoring ended. Final status: {counts}")
        held = counts.get(5, 0)
        if held > 0:
            held_jobs = self.get_held_reasons()
            print(f"  âŒ {held} jobs still held:")
            for j in held_jobs:
                print(f"    {j['name']}: {j['reason'][:120]}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 condor_submit_safe.py <submit_file> <campaign_dir>")
        print("  Runs preflight checks, submits, and monitors.")
        sys.exit(1)

    submit_file = sys.argv[1]
    campaign_dir = sys.argv[2]

    # 1. Parse run names
    names = parse_run_names(submit_file)
    if not names:
        print("âŒ No run names found in submit file")
        sys.exit(1)
    print(f"Found {len(names)} runs: {names}")

    # 2. Preflight
    pf = PreflightCheck(campaign_dir, names)
    ok = pf.run_all(submit_file)
    if not ok:
        print("\nâŒ Preflight failed. Fix issues and retry.")
        sys.exit(1)

    # 3. Submit
    print("\n=== Submitting ===")
    rc, out, err = _crc_ssh(f"cd {campaign_dir} && condor_submit {os.path.basename(submit_file)}")
    print(out)
    if rc != 0:
        print(f"âŒ Submit failed: {err}")
        sys.exit(1)

    # Parse cluster ID
    m = re.search(r"cluster (\d+)", out)
    if not m:
        print("âš ï¸  Could not parse cluster ID, skipping monitoring")
        sys.exit(0)
    cluster_id = int(m.group(1))
    print(f"Submitted as cluster {cluster_id}")

    # 4. Launch background monitor on CRC (nohup)
    # This runs independently on CRC, auto-fixes held jobs, logs to campaign dir
    monitor_script = "<BASELINE_HPC_SOURCE_ROOT>/scripts/auto_monitor.sh"
    print(f"\n=== Launching background monitor on CRC ===")
    rc, out = _crc_ssh(
        f"'nohup bash {monitor_script} {cluster_id} {campaign_dir} 120 60 "
        f"> {campaign_dir}/monitor_{cluster_id}.log 2>&1 &"
    )
    if rc == 0:
        print(f"  âœ… Monitor running on CRC (cluster {cluster_id})")
        print(f"  ðŸ“‹ Log: {campaign_dir}/monitor_{cluster_id}.log")
        print(f"  Checks every 2min, auto-fixes held jobs, stops when all done")
    else:
        print(f"  âš ï¸  Could not start monitor: {out}")
        print(f"  Falling back to local monitoring (5 polls Ã— 60s)")
        monitor = PostSubmitMonitor(cluster_id, campaign_dir, names, poll_interval=60, max_polls=5)
        monitor.run()


if __name__ == "__main__":
    main()



