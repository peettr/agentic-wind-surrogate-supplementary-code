"""Executor: CRC Condor batch submission + sentinel-driven monitoring.

Runs from **Windows**; all Condor commands are issued on CRC's ``<HPC_FILE_LOGIN>``
login node over SSH via WSL's ControlMaster socket. Files are staged on
``<HPC_PATH>`` (AFS-shared across CRC nodes) so Condor's
``transfer_input_files`` / ``transfer_output_files`` machinery is unused —
jobs read inputs and write outputs directly to shared storage (fix #6).

Responsibilities
----------------
* Render the Condor submit template for each :class:`ExperimentConfig`.
* Stage the submit file + train_config.json + any codegen ``script_path``
  onto CRC via SSH.
* Submit jobs with ``ssh <HPC_FILE_LOGIN> 'cd ... && condor_submit ...'``.
* Poll ``condor_q``; map Condor JobStatus codes (1=Idle, 2=Running,
  3=Removed, 4=Completed, 5=Held, 6=Transferring, 7=Suspended) to logical
  states, treating Removed as failed and Suspended as requeue-eligible
  (fix #8).
* Detect evictions via checkpoint presence and let ``train.py`` resume.
* Fail fast on SSH/submission errors (fix #5): if submission cannot be
  verified with a cluster id, the job is marked ``failed`` immediately.
* GPU ClassAd matches on ``Machine`` hostname, not ``CUDACapability``
  (fix #7), per CRC Dodi.
* AFS: Condor executors have no AFS token, so anything under
  ``/afs/...`` is unreachable. Use ``<HPC_USER_ROOT>/...`` paths.
  Run ``fs setacl /groups/... nd_campus read`` once for read permissions.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from string import Template
from typing import Optional

from shared.configs.schema import ExperimentConfig, JobHandle


LOGGER = logging.getLogger("auto_v3.executor")

DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "templates" / "condor_submit.template"
)
POLL_INTERVAL_SEC = 60
IDLE_TIMEOUT_SEC = 30 * 60
DEFAULT_SSH_TIMEOUT_SEC = 60
DEFAULT_SUBMIT_TIMEOUT_SEC = 120
# Safety cap on collect_results so a stuck remote job does not block the
# Windows runner forever. Well beyond the longest expected 500-epoch job.
DEFAULT_COLLECT_TIMEOUT_SEC = 24 * 3600

# Condor JobStatus codes (from the manual): 1=Idle, 2=Running, 3=Removed,
# 4=Completed, 5=Held, 6=Transferring, 7=Suspended. See fix #8.
_STATUS_MAP = {
    "1": "I",
    "2": "R",
    "3": "X",   # Removed -> evicted/failed
    "4": "C",
    "5": "H",
    "6": "T",   # Transferring output
    "7": "S",   # Suspended
}
_LOGICAL = {
    "I": "idle",
    "R": "running",
    "C": "completed",
    "H": "held",
    "X": "evicted",
    "T": "running",   # transferring output — treat as running
    "S": "idle",      # suspended — requeue-eligible, keep tracking
}


# ---------------------------------------------------------------------------
# SSH helpers (WSL ControlMaster)
# ---------------------------------------------------------------------------
@dataclass
class SSHSettings:
    """Configuration for WSL-relayed SSH to CRC.

    On Windows, we invoke ``wsl bash -lc "ssh ..."`` so that the
    ControlMaster socket at ``<SSH_CONTROL_PATH>
    (set up by the human researcher manually) is reused by every command.
    """

    user: str = "lhu1"
    host: str = "<HPC_FILE_LOGIN>"
    control_path: str = "<SSH_CONTROL_PATH>"
    connect_timeout: int = 30
    use_wsl: bool = True


def build_ssh_cmd(
    remote_cmd: str, settings: SSHSettings,
) -> list[str]:
    """Build the argv for a single remote command via WSL SSH ControlMaster.

    Uses ``wsl bash -lc "<cmd>"`` on Windows so the user's login shell
    (and therefore the ControlMaster socket) is active. The SSH options
    enforce non-interactive (BatchMode=yes) and a 30 s connect timeout
    (CRC login banner is long; anything shorter risks false failures).
    """
    ssh_cmd = (
        f"ssh -o BatchMode=yes "
        f"-o ConnectTimeout={settings.connect_timeout} "
        f"-o ControlPath={settings.control_path} "
        f"{settings.user}@{settings.host} "
        + shlex.quote(remote_cmd)
    )
    if settings.use_wsl:
        return ["wsl", "bash", "-lc", ssh_cmd]
    return ["bash", "-lc", ssh_cmd]


def build_scp_cmd(
    local_path: Path,
    remote_path: str,
    settings: SSHSettings,
) -> list[str]:
    """Build argv for an scp upload via WSL + ControlMaster."""
    # Translate Windows path for WSL if needed.
    local = str(local_path).replace("\\", "/")
    if settings.use_wsl and re.match(r"^[A-Za-z]:", local):
        drive = local[0].lower()
        local = f"/mnt/{drive}{local[2:]}"
    scp_cmd = (
        f"scp -o BatchMode=yes "
        f"-o ConnectTimeout={settings.connect_timeout} "
        f"-o ControlPath={settings.control_path} "
        f"{shlex.quote(local)} "
        f"{settings.user}@{settings.host}:{shlex.quote(remote_path)}"
    )
    if settings.use_wsl:
        return ["wsl", "bash", "-lc", scp_cmd]
    return ["bash", "-lc", scp_cmd]


# ---------------------------------------------------------------------------
# Remote filesystem layout
# ---------------------------------------------------------------------------
@dataclass
class RemoteLayout:
    """Absolute paths on CRC shared filesystem (``<HPC_USER_ROOT>/...``)."""

    project_root: str = "<BASELINE_HPC_SOURCE_ROOT>"
    data_dir: str = "<BASELINE_HPC_SOURCE_ROOT>/shared/data"
    split_manifest: str = (
        "<BASELINE_HPC_SOURCE_ROOT>/shared/data/split_manifest.json"
    )
    python: str = "python"  # CRC module already on PATH via getenv=True

    def submit_dir(self, campaign_id: str) -> str:
        return f"{self.project_root}/campaigns/{campaign_id}/submit"

    def runs_dir(self, campaign_id: str) -> str:
        return f"{self.project_root}/campaigns/{campaign_id}/runs"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
@dataclass
class RemoteResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str = ""


class Executor:
    """Submit and monitor CRC Condor GPU jobs over SSH."""

    def __init__(
        self,
        campaign_dir: str | Path,
        template_path: Optional[str | Path] = None,
        script_path: Optional[str | Path] = None,
        data_dir: str | Path = "shared/data",
        split_manifest_path: str | Path = "shared/data/split_manifest.json",
        ssh: Optional[SSHSettings] = None,
        remote: Optional[RemoteLayout] = None,
        campaign_id: Optional[str] = None,
        local_mode: Optional[bool] = None,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.template_path = Path(template_path or DEFAULT_TEMPLATE)
        self.script_path = (
            Path(script_path) if script_path
            else Path(__file__).resolve().parent.parent / "shared" / "train.py"
        )
        self.data_dir = Path(data_dir)
        self.split_manifest_path = Path(split_manifest_path)
        self.submit_dir = self.campaign_dir / "submit"
        self.results_root = self.campaign_dir / "runs"
        self.submit_dir.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.ssh = ssh or SSHSettings()
        self.remote = remote or RemoteLayout()
        self.campaign_id = campaign_id or self.campaign_dir.name
        # ``local_mode`` falls back to True iff we cannot find the WSL
        # binary AND cannot find condor_submit locally — this is mostly
        # useful for offline unit tests.
        self.local_mode = (
            local_mode
            if local_mode is not None
            else (shutil.which("wsl") is None and shutil.which("condor_submit") is None)
        )

    # ------------------------------------------------------------------
    # Handle reconstruction (for phase-based runner)
    # ------------------------------------------------------------------
    def reconstruct_handles(
        self, handle_info: list[dict], campaign_dir: Path,
    ) -> list[JobHandle]:
        """Reconstruct JobHandle objects from persisted handle metadata.

        The phase-based runner stores handle info in state.json between
        submit and collect phases. This method rebuilds the handles.
        """
        handles = []
        for info in handle_info:
            results_dir = str(campaign_dir / "runs" / info["experiment_id"])
            handles.append(JobHandle(
                experiment_id=info["experiment_id"],
                cluster_id=info.get("cluster_id"),
                results_dir=results_dir,
                remote_results_dir=info.get("remote_results_dir"),
                submit_file=str(campaign_dir / "submit" / f"{info['experiment_id']}.submit"),
                status="submitted",  # will be updated by poll()
            ))
        return handles

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit_batch(
        self, configs: list[ExperimentConfig],
    ) -> list[JobHandle]:
        return [self._submit_one(c) for c in configs]

    def _remote_results_dir(self, cfg: ExperimentConfig) -> str:
        return f"{self.remote.runs_dir(self.campaign_id)}/{cfg.experiment_id}"

    def _remote_submit_path(self, cfg: ExperimentConfig) -> str:
        return f"{self.remote.submit_dir(self.campaign_id)}/{cfg.experiment_id}.submit"

    def _remote_config_path(self, cfg: ExperimentConfig) -> str:
        return f"{self._remote_results_dir(cfg)}/train_config.json"

    def _remote_train_script(self, cfg: ExperimentConfig) -> str:
        return f"{self.remote.project_root}/shared/train.py"

    def _remote_codegen_path(self, cfg: ExperimentConfig) -> Optional[str]:
        if not cfg.script_path:
            return None
        # Place codegen-produced files alongside the run so the Condor
        # worker can import them via absolute path (fix #10).
        name = Path(cfg.script_path).name
        return f"{self._remote_results_dir(cfg)}/{name}"

    def _submit_one(self, cfg: ExperimentConfig) -> JobHandle:
        local_results_dir = self.results_root / cfg.experiment_id
        local_results_dir.mkdir(parents=True, exist_ok=True)

        remote_results_dir = self._remote_results_dir(cfg)
        remote_script_path: Optional[str] = None
        if cfg.script_path:
            remote_script_path = self._remote_codegen_path(cfg)

        eval_splits = ["val", "holdout"] if cfg.phase != "search" else ["val"]

        train_cfg = {
            "experiment_id": cfg.experiment_id,
            "strategy": cfg.strategy,
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "checkpoint_interval": 50,
            "arch_name": cfg.arch_name,
            "arch_kwargs": cfg.variant.get("arch_kwargs", {}),
            "loss_name": cfg.loss_name,
            "loss_kwargs": cfg.variant.get("loss_kwargs", {}),
            "data_dir": self.remote.data_dir,
            "results_dir": remote_results_dir,
            "split_manifest_path": self.remote.split_manifest,
            "heartbeat_interval_epochs": 10,
            "phase": cfg.phase,
            "eval_splits": eval_splits,
            "script_path": remote_script_path,
        }
        cfg_path = local_results_dir / "train_config.json"
        cfg_path.write_text(json.dumps(train_cfg, indent=2))

        template = self.template_path.read_text()
        submit_text = Template(template).safe_substitute({
            "JOB_NAME": cfg.experiment_id,
            "SCRIPT_PATH": self._remote_train_script(cfg),
            "CONFIG_PATH": self._remote_config_path(cfg),
            "RESULTS_DIR": remote_results_dir,
            "STRATEGY": cfg.strategy,
            "SEED": str(cfg.seed),
            "PYTHON": self.remote.python,
        })
        submit_path = self.submit_dir / f"{cfg.experiment_id}.submit"
        submit_path.write_text(submit_text)

        handle = JobHandle(
            experiment_id=cfg.experiment_id,
            cluster_id=None,
            submit_file=str(submit_path),
            results_dir=str(local_results_dir),
            remote_results_dir=(None if self.local_mode else remote_results_dir),
            status="submitted",
        )

        # Stage + submit.
        staged = self._stage_and_submit(
            cfg, cfg_path, submit_path, remote_results_dir,
        )
        if not staged.ok:
            # Fail fast (fix #5): no remote cluster id -> immediate failure.
            LOGGER.error(
                "Submission failed for %s: %s",
                cfg.experiment_id, staged.error,
            )
            handle.status = "failed"
            (local_results_dir / "FAILED").write_text(
                json.dumps(
                    {"error": staged.error, "stderr": staged.stderr}, indent=2,
                )
            )
            return handle
        handle.cluster_id = staged.stdout.strip() or None
        if handle.cluster_id is None:
            handle.status = "failed"
            (local_results_dir / "FAILED").write_text(
                json.dumps({"error": "no cluster_id parsed"}, indent=2)
            )
        return handle

    def _stage_and_submit(
        self,
        cfg: ExperimentConfig,
        cfg_path: Path,
        submit_path: Path,
        remote_results_dir: str,
    ) -> RemoteResult:
        if self.local_mode:
            return self._local_submit(submit_path)
        # 1. mkdir -p on remote.
        mkdir_cmd = (
            f"mkdir -p {shlex.quote(remote_results_dir)} "
            f"{shlex.quote(self.remote.submit_dir(self.campaign_id))}"
        )
        res = self._run_remote(mkdir_cmd, timeout=DEFAULT_SSH_TIMEOUT_SEC)
        if not res.ok:
            return res
        # 2. scp submit + config + (optional) codegen script.
        for local, remote in [
            (cfg_path, self._remote_config_path(cfg)),
            (submit_path, self._remote_submit_path(cfg)),
        ]:
            res = self._run_local(
                build_scp_cmd(local, remote, self.ssh),
                timeout=DEFAULT_SSH_TIMEOUT_SEC,
            )
            if not res.ok:
                return res
        if cfg.script_path:
            remote_codegen = self._remote_codegen_path(cfg)
            if cfg.script_path.startswith(self.remote.project_root):
                # script_path already lives on CRC (e.g. a generated-code
                # champion persisted from a prior run). Copy in place
                # remotely instead of trying to scp an imaginary local
                # file from the Windows runner (new issue 1 / partial 9).
                if cfg.script_path != remote_codegen:
                    res = self._run_remote(
                        f"cp {shlex.quote(cfg.script_path)} "
                        f"{shlex.quote(remote_codegen)}",
                        timeout=DEFAULT_SSH_TIMEOUT_SEC,
                    )
                else:
                    res = RemoteResult(ok=True)
            else:
                res = self._run_local(
                    build_scp_cmd(
                        Path(cfg.script_path), remote_codegen, self.ssh,
                    ),
                    timeout=DEFAULT_SSH_TIMEOUT_SEC,
                )
            if not res.ok:
                return res
        # 3. condor_submit on the remote.
        submit_cmd = (
            f"cd {shlex.quote(self.remote.project_root)} && "
            f"condor_submit {shlex.quote(self._remote_submit_path(cfg))}"
        )
        res = self._run_remote(submit_cmd, timeout=DEFAULT_SUBMIT_TIMEOUT_SEC)
        if not res.ok:
            return res
        m = re.search(r"cluster\s+(\d+)", res.stdout or "")
        if not m:
            return RemoteResult(
                ok=False,
                stdout=res.stdout,
                stderr=res.stderr,
                returncode=res.returncode,
                error="condor_submit output did not contain a cluster id",
            )
        return RemoteResult(
            ok=True, stdout=m.group(1), stderr=res.stderr, returncode=0,
        )

    def _local_submit(self, submit_path: Path) -> RemoteResult:
        exe = shutil.which("condor_submit")
        if exe is None:
            return RemoteResult(
                ok=False,
                error="condor_submit not on PATH (local_mode) — mark failed.",
            )
        try:
            r = subprocess.run(
                [exe, str(submit_path)],
                capture_output=True, text=True, check=False,
                timeout=DEFAULT_SUBMIT_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as exc:
            return RemoteResult(ok=False, error=f"condor_submit timed out: {exc}")
        m = re.search(r"cluster\s+(\d+)", r.stdout or "")
        if not m:
            return RemoteResult(
                ok=False, stdout=r.stdout, stderr=r.stderr,
                returncode=r.returncode,
                error="condor_submit output did not contain a cluster id",
            )
        return RemoteResult(
            ok=True, stdout=m.group(1), stderr=r.stderr, returncode=0,
        )

    # ------------------------------------------------------------------
    # Polling / collection
    # ------------------------------------------------------------------
    def poll(self, handles: list[JobHandle]) -> list[JobHandle]:
        statuses = self._condor_q_map(handles)
        for h in handles:
            if h.status == "failed":
                continue
            rd = Path(h.results_dir)
            code = statuses.get(h.cluster_id or "")
            if code is None:
                # Not in queue: consult local sentinels first, then remote
                # sentinels/metrics. In shared-filesystem mode the train.py
                # process writes FINISHED/metrics.json to
                # <HPC_PATH> on the remote host, so the Windows
                # runner has no local sentinel unless a separate sync mirror
                # exists (new issue 1 / partial 5 / partial 6).
                if (rd / "FINISHED").exists():
                    h.status = "completed"
                elif (rd / "FAILED").exists():
                    h.status = "failed"
                else:
                    remote_status = self._check_remote_sentinel(h)
                    if remote_status == "completed":
                        h.status = "completed"
                    elif remote_status == "failed":
                        h.status = "failed"
                    # else leave current status; may still be "submitted".
            else:
                logical = _LOGICAL.get(code, h.status)
                # Treat Removed/3 as an outright failure unless a
                # checkpoint exists (fix #8 — allow eviction recovery).
                if code == "X" and not (rd / "checkpoint.pt").exists():
                    h.status = "failed"
                    if not (rd / "FAILED").exists():
                        (rd / "FAILED").write_text(
                            json.dumps({"error": "condor removed job"}, indent=2)
                        )
                else:
                    h.status = logical
        return handles

    def _check_remote_sentinel(self, h: JobHandle) -> Optional[str]:
        """Probe the remote shared filesystem for a terminal sentinel.

        Returns ``"completed"`` if ``FINISHED`` or ``metrics.json`` exists
        on the remote host, ``"failed"`` if ``FAILED`` exists, or ``None``
        when there is no signal (or we're in local mode / lack a remote
        path / the SSH probe itself errors).
        """
        if self.local_mode or not h.remote_results_dir:
            return None
        rd = h.remote_results_dir
        # Single round-trip via a short shell snippet to avoid paying SSH
        # latency three times per job per poll.
        cmd = (
            f"if [ -f {shlex.quote(rd + '/FINISHED')} ]; then echo FINISHED; "
            f"elif [ -f {shlex.quote(rd + '/FAILED')} ]; then echo FAILED; "
            f"elif [ -f {shlex.quote(rd + '/metrics.json')} ]; "
            f"then echo METRICS; else echo NONE; fi"
        )
        res = self._run_remote(cmd, timeout=DEFAULT_SSH_TIMEOUT_SEC)
        if not res.ok:
            LOGGER.debug(
                "Remote sentinel probe failed for %s: %s",
                h.experiment_id, res.error or res.stderr,
            )
            return None
        token = (res.stdout or "").strip().splitlines()[-1:]
        marker = token[0] if token else ""
        if marker in ("FINISHED", "METRICS"):
            return "completed"
        if marker == "FAILED":
            return "failed"
        return None

    def collect_results(
        self,
        handles: list[JobHandle],
        poll_interval: int = POLL_INTERVAL_SEC,
        timeout: int = DEFAULT_COLLECT_TIMEOUT_SEC,
    ) -> list[dict]:
        """Block until every job reaches a terminal state; return metrics.

        An absolute ``timeout`` cap prevents the runner from hanging
        forever if remote sentinel detection fails (new issue 1). Jobs
        still pending when the cap fires are marked failed so the caller
        can record the outcome and move on.
        """
        idle_since: dict[str, float] = {}
        deadline = time.time() + max(timeout, 0)
        while True:
            self.poll(handles)
            pending = [
                h for h in handles
                if h.status not in ("completed", "failed")
            ]
            now = time.time()
            if now >= deadline and pending:
                LOGGER.error(
                    "collect_results timeout after %ds; %d job(s) still "
                    "pending — marking as failed.",
                    timeout, len(pending),
                )
                for h in pending:
                    h.status = "failed"
                break
            for h in pending:
                if h.status == "idle":
                    idle_since.setdefault(h.experiment_id, now)
                    if now - idle_since[h.experiment_id] > IDLE_TIMEOUT_SEC:
                        LOGGER.warning(
                            "Job %s idle > %ds; requeueing.",
                            h.experiment_id, IDLE_TIMEOUT_SEC,
                        )
                        self._requeue(h)
                        idle_since[h.experiment_id] = now
                elif h.status == "held":
                    LOGGER.warning(
                        "Job %s held; attempting release.", h.experiment_id,
                    )
                    self._release(h)
                elif h.status == "evicted":
                    LOGGER.info(
                        "Job %s evicted; will resume from checkpoint.",
                        h.experiment_id,
                    )
            if not pending:
                break
            time.sleep(poll_interval)

        out: list[dict] = []
        for h in handles:
            # Results written to the shared /groups mount on CRC; they are
            # visible via the same path the human researcher has mounted on the login
            # node. If the local sync mirrors the runs dir, metrics.json
            # will also be present locally. Prefer local; else try remote.
            mpath_local = Path(h.results_dir) / "metrics.json"
            payload: Optional[dict] = None
            if mpath_local.exists():
                try:
                    payload = json.loads(mpath_local.read_text())
                except json.JSONDecodeError as exc:
                    LOGGER.warning(
                        "Malformed metrics.json for %s: %s",
                        h.experiment_id, exc,
                    )
            if payload is None:
                payload = self._fetch_remote_metrics(h)
            if payload is None:
                payload = {
                    "experiment_id": h.experiment_id,
                    "strategy": "unknown",
                    "arch_name": "unknown",
                    "loss_name": "unknown",
                    "seed": 0,
                    "status": "failed",
                    "error_message": "metrics.json missing",
                    "val_metrics": {
                        "r2_median": float("nan"),
                        "r2_mean": float("nan"),
                    },
                }
            out.append(payload)
        return out

    def _fetch_remote_metrics(self, h: JobHandle) -> Optional[dict]:
        """Best-effort ssh cat of metrics.json from the shared filesystem."""
        if self.local_mode:
            return None
        # Prefer the handle's tracked remote path (stored at submit time);
        # fall back to recomputing it for legacy handles persisted without
        # ``remote_results_dir`` (partial 6).
        remote_dir = h.remote_results_dir or str(
            PurePosixPath(self.remote.runs_dir(self.campaign_id))
            / h.experiment_id
        )
        remote_path = f"{remote_dir}/metrics.json"
        cmd = f"cat {shlex.quote(remote_path)}"
        res = self._run_remote(cmd, timeout=DEFAULT_SSH_TIMEOUT_SEC)
        if not res.ok or not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def cancel(self, handles: list[JobHandle]) -> None:
        for h in handles:
            if not h.cluster_id:
                continue
            if self.local_mode:
                exe = shutil.which("condor_rm")
                if exe is not None:
                    subprocess.run([exe, h.cluster_id], check=False)
            else:
                self._run_remote(
                    f"condor_rm {shlex.quote(h.cluster_id)}",
                    timeout=DEFAULT_SSH_TIMEOUT_SEC,
                )

    # ------------------------------------------------------------------
    # Condor wrappers
    # ------------------------------------------------------------------
    def _condor_q_map(
        self, handles: Optional[list[JobHandle]] = None,
    ) -> dict[str, str]:
        ids = [h.cluster_id for h in (handles or []) if h.cluster_id]
        if self.local_mode:
            exe = shutil.which("condor_q")
            if exe is None:
                return {}
            cmd = [
                exe, "-nobatch",
                "-format", "%d.%d ", "ClusterId",
                "-format", "%d ", "ProcId",
                "-format", "%s\n", "JobStatus",
            ]
            try:
                res = subprocess.run(
                    cmd, capture_output=True, text=True,
                    check=False, timeout=DEFAULT_SSH_TIMEOUT_SEC,
                )
            except subprocess.TimeoutExpired:
                LOGGER.warning("condor_q timed out")
                return {}
            return self._parse_condor_q(res.stdout)
        # Remote: constrain to our clusters if we know them.
        constraint = ""
        if ids:
            joined = " || ".join(f"ClusterId=={i}" for i in ids)
            constraint = f" -constraint {shlex.quote(joined)}"
        cmd = (
            f"condor_q{constraint} -nobatch "
            f"-format '%d.%d ' ClusterId -format '%d ' ProcId "
            f"-format '%s\\n' JobStatus"
        )
        res = self._run_remote(cmd, timeout=DEFAULT_SSH_TIMEOUT_SEC)
        if not res.ok:
            LOGGER.warning("Remote condor_q failed: %s", res.error or res.stderr)
            return {}
        return self._parse_condor_q(res.stdout)

    @staticmethod
    def _parse_condor_q(stdout: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in (stdout or "").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                cid = parts[0].split(".")[0]
                out[cid] = _STATUS_MAP.get(parts[-1], "I")
        return out

    def _release(self, h: JobHandle) -> None:
        if not h.cluster_id:
            return
        if self.local_mode:
            exe = shutil.which("condor_release")
            if exe is not None:
                subprocess.run([exe, h.cluster_id], check=False)
            return
        self._run_remote(
            f"condor_release {shlex.quote(h.cluster_id)}",
            timeout=DEFAULT_SSH_TIMEOUT_SEC,
        )

    def _requeue(self, h: JobHandle) -> None:
        if h.cluster_id:
            if self.local_mode:
                if shutil.which("condor_rm"):
                    subprocess.run(
                        ["condor_rm", h.cluster_id], check=False,
                    )
            else:
                self._run_remote(
                    f"condor_rm {shlex.quote(h.cluster_id)}",
                    timeout=DEFAULT_SSH_TIMEOUT_SEC,
                )
        # Resubmit the same submit file.
        if self.local_mode:
            res = self._local_submit(Path(h.submit_file))
        else:
            submit_cmd = (
                f"cd {shlex.quote(self.remote.project_root)} && "
                f"condor_submit {shlex.quote(self._posixify(h.submit_file))}"
            )
            res = self._run_remote(submit_cmd, timeout=DEFAULT_SUBMIT_TIMEOUT_SEC)
            if res.ok:
                m = re.search(r"cluster\s+(\d+)", res.stdout or "")
                if m:
                    res = RemoteResult(
                        ok=True, stdout=m.group(1),
                        stderr=res.stderr, returncode=0,
                    )
                else:
                    res = RemoteResult(
                        ok=False,
                        error="condor_submit output did not contain a cluster id",
                    )
        if res.ok:
            h.cluster_id = res.stdout.strip() or None
            h.status = "submitted"
        else:
            LOGGER.error("Requeue failed for %s: %s", h.experiment_id, res.error)
            h.status = "failed"

    def _posixify(self, local_submit_path: str) -> str:
        """Convert local submit file path to the matching path on CRC."""
        name = Path(local_submit_path).name
        return f"{self.remote.submit_dir(self.campaign_id)}/{name}"

    # ------------------------------------------------------------------
    # Low-level command runners
    # ------------------------------------------------------------------
    @staticmethod
    def _run_local(argv: list[str], timeout: int) -> RemoteResult:
        try:
            r = subprocess.run(
                argv, capture_output=True, text=True,
                check=False, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return RemoteResult(ok=False, error=f"timeout: {exc}")
        except FileNotFoundError as exc:
            return RemoteResult(ok=False, error=f"binary missing: {exc}")
        if r.returncode != 0:
            return RemoteResult(
                ok=False, stdout=r.stdout or "", stderr=r.stderr or "",
                returncode=r.returncode,
                error=f"rc={r.returncode}: {r.stderr.strip()[:200]}",
            )
        return RemoteResult(
            ok=True, stdout=r.stdout or "", stderr=r.stderr or "",
            returncode=0,
        )

    def _run_remote(self, remote_cmd: str, timeout: int) -> RemoteResult:
        argv = build_ssh_cmd(remote_cmd, self.ssh)
        return self._run_local(argv, timeout=timeout)


__all__ = ["Executor", "SSHSettings", "RemoteLayout", "build_ssh_cmd"]
