"""Sequential Workflow Executor â€” Submit/monitor jobs on CRC via SSH.

Three submission paths:
1. GPU training (external scheduler): uses scheduler_submit.template, shared filesystem
2. CPU analysis (SGE qsub script): metrics aggregation, eval, etc.
3. One-off commands (SGE qsub): lightweight CRC-side operations

All remote commands use WSL SSH ControlMaster to <HPC_FILE_LOGIN>.
Shared filesystem: <PROJECT_HPC_ROOT>/
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import base64
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from string import Template
from typing import Optional

LOGGER = logging.getLogger("hybrid.executor")

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "models" / "generated"
SHARED_MODELS_DIR = PROJECT_ROOT / "shared" / "models"
external_scheduler_TEMPLATE = PROJECT_ROOT / "templates" / "scheduler_submit.template"
external_scheduler_WRAPPER = PROJECT_ROOT / "templates" / "external_scheduler_wrapper.sh"

DEFAULT_SSH_TIMEOUT = 420
DEFAULT_SUBMIT_TIMEOUT = 480
POLL_INTERVAL_SEC = 60

# external scheduler JobStatus codes
_external_scheduler_STATUS = {
    "1": "I",  # Idle
    "2": "R",  # Running
    "3": "X",  # Removed
    "4": "C",  # Completed
    "5": "H",  # Held
    "6": "T",  # Transferring
    "7": "S",  # Suspended
}
_external_scheduler_LOGICAL = {
    "I": "idle", "R": "running", "C": "completed",
    "H": "held", "X": "evicted", "T": "running", "S": "idle",
}


# â”€â”€ SSH / Remote config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class SSHSettings:
    user: str = "lhu1"
    host: str = "<HPC_LOGIN>"
    control_path: str = "<SSH_CONTROL_PATH>"
    connect_timeout: int = 60
    use_wsl: bool = True


@dataclass
class RemoteLayout:
    """Absolute paths on CRC shared filesystem."""
    project_root: str = "<HYBRID_HPC_SOURCE_ROOT>"
    shared_root: str = "<PROJECT_HPC_ROOT>"
    train_script: str = "<HYBRID_HPC_SOURCE_ROOT>/shared/train.py"
    data_dir: str = "<BASELINE_HPC_SOURCE_ROOT>/shared/data"
    split_manifest: str = "<BASELINE_HPC_SOURCE_ROOT>/shared/data/split_manifest.json"

    def campaign_root(self, campaign_id: str) -> str:
        return f"{self.project_root}/campaigns/{campaign_id}"

    def runs_dir(self, campaign_id: str) -> str:
        return f"{self.campaign_root(campaign_id)}/runs"

    def submit_dir(self, campaign_id: str) -> str:
        return f"{self.campaign_root(campaign_id)}/submit"

    def analysis_dir(self, campaign_id: str) -> str:
        return f"{self.campaign_root(campaign_id)}/analysis"


# â”€â”€ Result types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class RemoteResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error: str = ""


@dataclass
class JobHandle:
    experiment_id: str
    cluster_id: Optional[str] = None
    scheduler: str = "external scheduler"  # "external scheduler" or "sge"
    submit_file: str = ""
    results_dir: str = ""
    remote_results_dir: str = ""
    status: str = "submitted"
    job_name: str = ""
    log_tail: str = ""
    error: str = ""
    submitted_at: float = 0.0  # timestamp when first submitted


# â”€â”€ SSH helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_ssh_cmd(remote_cmd: str, ssh: SSHSettings) -> list[str]:
    ssh_cmd = (
        f"ssh -o BatchMode=yes "
        f"-o ConnectTimeout={ssh.connect_timeout} "
        f"-o ControlPath={ssh.control_path} "
        f"{ssh.user}@{ssh.host} "
        + shlex.quote(remote_cmd)
    )
    if ssh.use_wsl:
        return ["wsl", "bash", "-lc", ssh_cmd]
    return ["bash", "-lc", ssh_cmd]


def build_scp_cmd(local_path: Path, remote_path: str, ssh: SSHSettings) -> list[str]:
    local = str(local_path).replace("\\", "/")
    if ssh.use_wsl and re.match(r"^[A-Za-z]:", local):
        drive = local[0].lower()
        local = f"/mnt/{drive}{local[2:]}"
    scp_cmd = (
        f"scp -o BatchMode=yes "
        f"-o ConnectTimeout={ssh.connect_timeout} "
        f"-o ControlPath={ssh.control_path} "
        f"{shlex.quote(local)} "
        f"{ssh.user}@{ssh.host}:{shlex.quote(remote_path)}"
    )
    if ssh.use_wsl:
        return ["wsl", "bash", "-lc", scp_cmd]
    return ["bash", "-lc", scp_cmd]


def build_stream_upload_cmd(local_path: Path, remote_path: str, ssh: SSHSettings) -> list[str]:
    """Upload via `base64 | ssh base64 -d`, avoiding flaky scp/SFTP hangs."""
    local = str(local_path).replace("\\", "/")
    if ssh.use_wsl and re.match(r"^[A-Za-z]:", local):
        drive = local[0].lower()
        local = f"/mnt/{drive}{local[2:]}"
    remote_dir = str(PurePosixPath(remote_path).parent)
    remote_cmd = f"mkdir -p {shlex.quote(remote_dir)} && base64 -d > {shlex.quote(remote_path)}"
    ssh_cmd = (
        f"base64 -w 0 {shlex.quote(local)} | "
        f"ssh -o BatchMode=yes "
        f"-o ConnectTimeout={ssh.connect_timeout} "
        f"-o ControlPath={ssh.control_path} "
        f"{ssh.user}@{ssh.host} "
        f"{shlex.quote(remote_cmd)}"
    )
    if ssh.use_wsl:
        return ["wsl", "bash", "-lc", ssh_cmd]
    return ["bash", "-lc", ssh_cmd]


def run_local(argv: list[str], timeout: int, retry: int = 1) -> RemoteResult:
    for attempt in range(1 + retry):
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    p.kill()
                try:
                    stdout, stderr = p.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                return RemoteResult(ok=False, stdout=stdout or "", stderr=stderr or "",
                                    error=f"timeout after {timeout}s: {exc}")
        except subprocess.TimeoutExpired as exc:
            return RemoteResult(ok=False, error=f"timeout: {exc}")
        except FileNotFoundError as exc:
            return RemoteResult(ok=False, error=f"binary missing: {exc}")
        rc = p.returncode or 0
        if rc == 0:
            return RemoteResult(ok=True, stdout=stdout or "", stderr=stderr or "", returncode=0)
        stderr_text = (stderr or "").strip()
        if "Permission denied" in stderr_text and attempt < retry:
            LOGGER.warning("SSH permission denied, retrying (attempt %d/%d)", attempt + 1, retry + 1)
            time.sleep(2)
            continue
        return RemoteResult(
            ok=False, stdout=stdout or "", stderr=stderr or "",
            returncode=rc,
            error=f"rc={rc}: {stderr_text[:200]}",
        )
    return RemoteResult(ok=False, error="max retries exceeded")


# â”€â”€ Executor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _registry_class_for_arch(arch_name: str) -> str | None:
    """Return class name registered for an arch in shared/models/__init__.py."""
    try:
        text = (SHARED_MODELS_DIR / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return None
    # Direct class registration: REGISTRY.register("arch", ClassName)
    m = re.search(
        rf"REGISTRY\.register\(\s*['\"]{re.escape(arch_name)}['\"]\s*,\s*([A-Za-z_]\w*)\s*\)",
        text,
    )
    if m:
        return m.group(1)
    # Lambda registration: REGISTRY.register("arch", lambda **kw: ClassName(...))
    m = re.search(
        rf"REGISTRY\.register\(\s*['\"]{re.escape(arch_name)}['\"]\s*,\s*lambda[^:]*:\s*([A-Za-z_]\w*)",
        text,
    )
    return m.group(1) if m else None


def _make_run_local_model_code(code: str, arch_name: str = "") -> str:
    """Make a reference model importable as standalone run-local model.py."""
    code = re.sub(r"^\s*from\s+\.base\s+import\s+BaseSurrogate\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from\s+\.[\w_]+\s+import\s+.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"\(\s*BaseSurrogate\s*\)", "(nn.Module)", code)
    if "import torch.nn as nn" not in code and "import torch.nn" not in code:
        code = "import torch\nimport torch.nn as nn\n\n" + code
    elif "import torch" not in code:
        code = "import torch\n" + code
    # shared/train.py resolves external models by exact cfg.arch_name. Reference
    # classes are often CamelCase/uppercase (e.g. UNO), while arch_name is the
    # registry key (e.g. uno). Add a wrapper class, not a plain alias, so common
    # TrainConfig kwargs (n_c/depth/in_channels) can be translated to reference
    # constructor names such as base_ch.
    if arch_name and not re.search(rf"^\s*class\s+{re.escape(arch_name)}\b", code, flags=re.MULTILINE):
        class_names = re.findall(r"^class\s+([A-Za-z_]\w*)\s*\(", code, flags=re.MULTILINE)
        registered = _registry_class_for_arch(arch_name)
        preferred = registered if registered in class_names else None
        if preferred is None:
            preferred = next((c for c in class_names if c.lower() == arch_name.lower()), None)
        if preferred is None:
            helper_names = {"DoubleConv", "ConvBlock", "Down", "Up", "ResBlock", "Block", "Encoder", "Decoder"}
            non_helpers = [c for c in class_names if c not in helper_names]
            if len(non_helpers) == 1:
                preferred = non_helpers[0]
            elif len(class_names) == 1:
                preferred = class_names[0]
        if preferred:
            code = code.rstrip() + f'''

# Wrapper for Hybrid script_path loading by registry arch_name.
class {arch_name}({preferred}):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, **kwargs):
        import inspect
        sig = inspect.signature({preferred}.__init__)
        call_kwargs = {{}}
        if "in_channels" in sig.parameters:
            call_kwargs["in_channels"] = in_channels
        if "out_channels" in sig.parameters:
            call_kwargs["out_channels"] = out_channels
        if "n_c" in sig.parameters:
            call_kwargs["n_c"] = n_c
        if "base_ch" in sig.parameters:
            call_kwargs["base_ch"] = n_c
        if "depth" in sig.parameters:
            call_kwargs["depth"] = depth
        for _k, _v in kwargs.items():
            if _k in sig.parameters:
                call_kwargs[_k] = _v
        super().__init__(**call_kwargs)
'''
    return code


class Executor:
    """CRC job submission and monitoring."""

    def __init__(
        self,
        campaign_dir: Path,
        ssh: Optional[SSHSettings] = None,
        remote: Optional[RemoteLayout] = None,
    ) -> None:
        self.campaign_dir = campaign_dir
        self.campaign_id = campaign_dir.name
        self.ssh = ssh or SSHSettings()
        self.remote = remote or RemoteLayout()
        self.local_mode = shutil.which("wsl") is None

    # â”€â”€ Low-level remote execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_remote(self, remote_cmd: str, timeout: int = DEFAULT_SSH_TIMEOUT) -> RemoteResult:
        if self.local_mode:
            return RemoteResult(ok=False, error="local mode, no SSH")
        argv = build_ssh_cmd(remote_cmd, self.ssh)
        return run_local(argv, timeout=timeout)

    def scp_upload(self, local_path: Path, remote_path: str, timeout: int = 240) -> RemoteResult:
        if self.local_mode:
            return RemoteResult(ok=False, error="local mode, no SCP")
        data = local_path.read_bytes()
        if len(data) <= 200_000:
            # Avoid ssh stdin/pipe hangs observed with ControlMaster on CRC by
            # embedding small generated configs/submit/model files directly in
            # the remote command.
            encoded = base64.b64encode(data).decode("ascii")
            remote_dir = str(PurePosixPath(remote_path).parent)
            cmd = (
                f"mkdir -p {shlex.quote(remote_dir)} && "
                f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_path)}"
            )
            return self.run_remote(cmd, timeout=timeout)
        argv = build_stream_upload_cmd(local_path, remote_path, self.ssh)
        return run_local(argv, timeout=timeout)

    def ssh_ok(self) -> tuple[bool, str]:
        """Check SSH via existing ControlMaster socket. Does NOT attempt reconnect."""
        # CRC login/banner and ControlMaster reuse can be slow. Short
        # 10-15s probes can misclassify CRC as down; the human researcher confirmed 240s
        # is acceptable as the runner-level bound before human intervention.
        old_timeout = self.ssh.connect_timeout
        self.ssh.connect_timeout = 60
        try:
            r = self.run_remote("echo ok", timeout=420)
        finally:
            self.ssh.connect_timeout = old_timeout
        out = (r.stdout or "").strip()
        last = out.splitlines()[-1].strip() if out.splitlines() else ""
        if r.ok and last == "ok":
            return True, last
        return False, f"SSH failed: {(r.stderr or r.error or '')[:200]}"

    # â”€â”€ Path 1: GPU training (external scheduler) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def submit_gpu_train(
        self,
        exp_id: str,
        train_config: dict,
        gpu_requirements: str = '(regexp("qa-h100-", Machine) || regexp("qa-a100-", Machine) || regexp("qa-l40s-", Machine) || regexp("qa-a40-", Machine) || regexp("ta-a6k-", Machine)) && !regexp("qa-a40-003", Machine)',
        request_memory_gb: int = 16,
    ) -> JobHandle:
        """Submit a GPU training job via external scheduler (V3 pattern)."""
        remote_results = f"{self.remote.runs_dir(self.campaign_id)}/{exp_id}"
        remote_submit = f"{self.remote.submit_dir(self.campaign_id)}/{exp_id}.submit"
        remote_config = f"{remote_results}/train_config.json"
        local_results = self.campaign_dir / "runs" / exp_id
        local_results.mkdir(parents=True, exist_ok=True)

        # Resolve the per-run model implementation before writing config.
        # Hard rule: training jobs may share train.py, losses, eval, and data,
        # but the model implementation itself must be a run-local file.  The
        # runner supplies a local/reference script_path; the executor uploads it
        # to runs/<exp_id>/model.py and rewrites TrainConfig.script_path to that
        # immutable attempt-local path.
        remote_cfg = dict(train_config)
        remote_cfg["experiment_id"] = exp_id
        remote_cfg["results_dir"] = remote_results
        # Remove top-level fields that TrainConfig doesn't accept
        # (n_c, depth belong in arch_kwargs only)
        remote_cfg.pop("n_c", None)
        remote_cfg.pop("depth", None)
        remote_cfg.pop("use_ema", None)
        remote_cfg.pop("ema_decay", None)
        remote_cfg.pop("model_source_kind", None)
        local_script_value = remote_cfg.get("script_path")
        if not local_script_value:
            handle = JobHandle(
                experiment_id=exp_id,
                scheduler="external scheduler",
                results_dir=str(local_results),
                remote_results_dir=remote_results,
                job_name=exp_id,
                status="failed",
                submitted_at=time.time(),
                error="missing script_path; per-run model isolation requires a model file",
            )
            (local_results / "FAILED").write_text(json.dumps({"error": handle.error}, indent=2))
            return handle
        local_model = Path(str(local_script_value))
        if not local_model.is_absolute():
            candidates = [
                GENERATED_DIR / local_model.name,
                PROJECT_ROOT / "shared" / "models" / local_model.name,
                PROJECT_ROOT / local_model,
            ]
            local_model = next((p for p in candidates if p.exists()), local_model)
        if not local_model.exists() or not local_model.is_file():
            handle = JobHandle(
                experiment_id=exp_id,
                scheduler="external scheduler",
                results_dir=str(local_results),
                remote_results_dir=remote_results,
                job_name=exp_id,
                status="failed",
                submitted_at=time.time(),
                error=f"model source not found for per-run isolation: {local_script_value}",
            )
            (local_results / "FAILED").write_text(json.dumps({"error": handle.error}, indent=2))
            return handle
        remote_model = f"{remote_results}/model.py"
        remote_cfg["script_path"] = remote_model

        # Materialize the exact model implementation that will be uploaded for
        # this attempt. Reference models are converted to standalone files so
        # run-local loading does not depend on shared/models package imports.
        upload_model = local_results / "model.py"
        model_code = local_model.read_text(encoding="utf-8")
        source_kind = train_config.get("model_source_kind", "unknown")
        try:
            is_reference = SHARED_MODELS_DIR.resolve() in local_model.resolve().parents or local_model.resolve().parent == SHARED_MODELS_DIR.resolve()
        except OSError:
            is_reference = source_kind == "reference_copy"
        if is_reference or source_kind == "reference_copy":
            model_code = _make_run_local_model_code(model_code, str(remote_cfg.get("arch_name", "")))
        upload_model.write_text(model_code, encoding="utf-8")

        # Write train_config.json locally
        cfg_path = local_results / "train_config.json"
        cfg_path.write_text(json.dumps(remote_cfg, indent=2), encoding="utf-8")

        # Render external scheduler submit template
        template_text = external_scheduler_TEMPLATE.read_text(encoding="utf-8")
        submit_text = Template(template_text).safe_substitute({
            "JOB_NAME": exp_id,
            "SCRIPT_PATH": self.remote.train_script,
            "CONFIG_PATH": remote_config,
            "RESULTS_DIR": remote_results,
            "STRATEGY": train_config.get("strategy", "Sequential"),
            "SEED": str(train_config.get("seed", 1)),
            "PYTHON": "python",
            "REQUEST_MEMORY_GB": str(request_memory_gb),
        })
        # Override GPU requirements if needed
        if gpu_requirements:
            submit_text = re.sub(
                r'^requirements\s*=.*$',
                f'requirements            = ({gpu_requirements})',
                submit_text,
                flags=re.MULTILINE,
            )
        submit_path = self.campaign_dir / "submit" / f"{exp_id}.submit"
        submit_path.parent.mkdir(parents=True, exist_ok=True)
        submit_path.write_text(submit_text, encoding="utf-8")

        handle = JobHandle(
            experiment_id=exp_id,
            scheduler="external scheduler",
            submit_file=str(submit_path),
            results_dir=str(local_results),
            remote_results_dir=remote_results,
            job_name=exp_id,
            submitted_at=time.time(),
        )

        # Idempotency guard: if a previous runner died after scheduler_submit but
        # before saving state, the job may already exist in the queue. Reuse it
        # instead of creating duplicate training jobs for the same experiment.
        # Keep the remote scheduler_queue itself bounded.  CRC scheduler_queue can occasionally
        # hang behind the SSH ControlMaster, and a stuck idempotency probe should
        # not block the whole autonomous workflow.
        batch_constraint = shlex.quote('JobBatchName == "' + exp_id + '"')
        check_cmd = (
            "timeout 360s scheduler_queue lhu1 -nobatch "
            f"-constraint {batch_constraint} "
            "-format '%d ' ClusterId || true"
        )
        existing = self.run_remote(check_cmd, timeout=420)
        existing_cluster = (existing.stdout or "").strip().split()[:1] if existing.ok else []
        if existing_cluster and existing_cluster[0].isdigit():
            handle.cluster_id = existing_cluster[0]
            handle.status = "submitted"
            LOGGER.info("external scheduler job already exists for %s -> cluster %s", exp_id, handle.cluster_id)
            return handle

        # Stage + submit
        # 1. mkdir remote
        res = self.run_remote(
            f"mkdir -p {shlex.quote(remote_results)} {shlex.quote(self.remote.submit_dir(self.campaign_id))}"
        )
        if not res.ok:
            handle.status = "failed"
            return handle

        # 2. SCP config + submit file + run-local model.py
        for local, remote in [(cfg_path, remote_config), (submit_path, remote_submit), (upload_model, remote_model)]:
            res = self.scp_upload(local, remote)
            if not res.ok:
                handle.status = "failed"
                (local_results / "FAILED").write_text(
                    json.dumps({"error": f"scp failed: {res.error}"}, indent=2))
                return handle

        # Keep local sidecar metadata outside TrainConfig, because Pydantic
        # rejects extra fields.  This records the model isolation provenance.
        import hashlib
        model_sha256 = hashlib.sha256(upload_model.read_bytes()).hexdigest()
        (local_results / "attempt_metadata.json").write_text(json.dumps({
            "model_source_local": str(local_model),
            "model_source_kind": source_kind,
            "model_sha256": model_sha256,
            "remote_model": remote_model,
        }, indent=2), encoding="utf-8")
        self.scp_upload(local_results / "attempt_metadata.json", f"{remote_results}/attempt_metadata.json")

        # Shared filesystem visibility barrier.  external scheduler jobs can start within a
        # few seconds on idle GPU slots; make sure the just-uploaded config and
        # submit file are visible before handing the job to the scheduler.
        required_paths = [remote_config, remote_submit, remote_model]
        tests = " && ".join(f"test -s {shlex.quote(p)}" for p in required_paths)
        # Keep this shell snippet deliberately simple.  Some CRC SSH command
        # paths have been fragile with command substitution/newlines, so avoid
        # `for i in $(seq ...)` here.
        barrier_cmd = f"{tests} && sync && sleep 8"
        res = self.run_remote(barrier_cmd, timeout=420)
        if not res.ok:
            handle.status = "failed"
            (local_results / "FAILED").write_text(
                json.dumps({"error": f"remote file visibility barrier failed: {res.error}"}, indent=2))
            return handle

        # 4. scheduler_submit (must activate conda first so getenv=True passes conda env)
        conda_activate = (
            "source /opt/crc/Modules/current/init/bash && "
            "module load conda/25.9.1 && "
            "source /software/c/conda/25.9.1/etc/profile.d/conda.sh && "
            "conda activate graphwind"
        )
        submit_cmd = (
            f"{conda_activate} && "
            f"cd {shlex.quote(self.remote.shared_root)} && "
            f"scheduler_submit {shlex.quote(remote_submit)}"
        )
        res = self.run_remote(submit_cmd, timeout=DEFAULT_SUBMIT_TIMEOUT)
        if not res.ok:
            handle.status = "failed"
            (local_results / "FAILED").write_text(
                json.dumps({"error": f"scheduler_submit failed: {res.error}"}, indent=2))
            return handle

        m = re.search(r"cluster\s+(\d+)", res.stdout or "")
        if not m:
            handle.status = "failed"
            (local_results / "FAILED").write_text(
                json.dumps({"error": "no cluster_id in scheduler_submit output"}, indent=2))
            return handle

        handle.cluster_id = m.group(1)
        LOGGER.info("external scheduler submitted %s -> cluster %s", exp_id, handle.cluster_id)
        return handle

    # â”€â”€ Path 2: CPU analysis (SGE qsub script) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def submit_cpu_task(
        self,
        task_name: str,
        python_script: str,
        args: str = "",
        working_dir: Optional[str] = None,
    ) -> JobHandle:
        """Submit a CPU analysis job via SGE qsub."""
        wd = working_dir or self.remote.campaign_root(self.campaign_id)
        exp_id = f"cpu_{task_name}_{int(time.time()) % 100000}"
        remote_results = f"{self.remote.analysis_dir(self.campaign_id)}"
        log_path = f"{remote_results}/{exp_id}"

        shell_content = (
            "#!/bin/bash\n"
            f"#$ -N {exp_id}\n"
            "#$ -cwd\n"
            "#$ -l h_vmem=8G\n"
            f"#$ -o {log_path}.out\n"
            f"#$ -e {log_path}.err\n"
            "source /opt/crc/Modules/current/init/bash\n"
            "module load conda/25.9.1\n"
            "source /software/c/conda/25.9.1/etc/profile.d/conda.sh\n"
            "conda activate graphwind\n"
            f"cd {wd}\n"
            f"python {python_script} {args}\n"
        )

        # Write shell locally, SCP, qsub
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8') as f:
            f.write(shell_content)
            tmp_shell = Path(f.name)

        try:
            remote_shell_dir = f"{remote_results}/shells"
            remote_shell = f"{remote_shell_dir}/{exp_id}.sh"
            self.run_remote(f"mkdir -p {shlex.quote(remote_shell_dir)}")
            res = self.scp_upload(tmp_shell, remote_shell)
            if not res.ok:
                return JobHandle(experiment_id=exp_id, scheduler="sge", status="failed",
                                 job_name=exp_id, error=res.error)

            # qsub
            qsub_cmd = f"cd {shlex.quote(wd)} && qsub {shlex.quote(remote_shell)}"
            res = self.run_remote(qsub_cmd, timeout=DEFAULT_SSH_TIMEOUT)
            job_id = ""
            if res.ok:
                for token in (res.stdout or "").strip().split():
                    if token.isdigit():
                        job_id = token
                        break

            handle = JobHandle(
                experiment_id=exp_id,
                cluster_id=job_id or None,
                scheduler="sge",
                results_dir=remote_results,
                remote_results_dir=remote_results,
                status="submitted" if job_id else "failed",
                job_name=exp_id,
            )
            return handle
        finally:
            tmp_shell.unlink(missing_ok=True)

    # â”€â”€ Path 3: One-off qsub command â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def submit_qsub_command(
        self,
        command: str,
        task_name: str = "v4cmd",
        working_dir: Optional[str] = None,
    ) -> JobHandle:
        """Submit a one-off command via SGE qsub (lightweight, fast)."""
        wd = working_dir or self.remote.shared_root
        exp_id = f"cmd_{task_name}_{int(time.time()) % 100000}"
        remote_results = f"{self.remote.analysis_dir(self.campaign_id)}"
        log_path = f"{remote_results}/{exp_id}"

        shell_content = (
            "#!/bin/bash\n"
            f"#$ -N {exp_id}\n"
            "#$ -cwd\n"
            "#$ -l h_vmem=4G\n"
            f"#$ -o {log_path}.out\n"
            f"#$ -e {log_path}.err\n"
            "source /opt/crc/Modules/current/init/bash\n"
            "module load conda/25.9.1\n"
            "source /software/c/conda/25.9.1/etc/profile.d/conda.sh\n"
            "conda activate graphwind\n"
            f"cd {wd}\n"
            f"{command}\n"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8') as f:
            f.write(shell_content)
            tmp_shell = Path(f.name)

        try:
            remote_shell_dir = f"{remote_results}/shells"
            remote_shell = f"{remote_shell_dir}/{exp_id}.sh"
            self.run_remote(f"mkdir -p {shlex.quote(remote_shell_dir)}")
            self.scp_upload(tmp_shell, remote_shell)

            qsub_cmd = f"cd {shlex.quote(wd)} && qsub {shlex.quote(remote_shell)}"
            res = self.run_remote(qsub_cmd, timeout=DEFAULT_SSH_TIMEOUT)
            job_id = ""
            if res.ok:
                for token in (res.stdout or "").strip().split():
                    if token.isdigit():
                        job_id = token
                        break

            return JobHandle(
                experiment_id=exp_id,
                cluster_id=job_id or None,
                scheduler="sge",
                results_dir=remote_results,
                remote_results_dir=remote_results,
                status="submitted" if job_id else "failed",
                job_name=exp_id,
            )
        finally:
            tmp_shell.unlink(missing_ok=True)

    # â”€â”€ Monitoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def poll_external_scheduler(self, handles: list[JobHandle]) -> dict[str, str]:
        """Poll external scheduler job status. Returns {cluster_id: status_code}."""
        ids = [h.cluster_id for h in handles if h.cluster_id and h.scheduler == "external scheduler"]
        if not ids:
            return {}
        constraint = " || ".join(f"ClusterId=={i}" for i in ids)
        # Bound the remote scheduler_queue call itself. If CRC's scheduler query or
        # the ControlMaster path hangs, return a transient empty status and let
        # the next tick retry instead of wedging monitor and leaving a stale lock.
        cmd = (
            f"timeout 360s scheduler_queue -constraint {shlex.quote(constraint)} -nobatch "
            f"-format '%d.%d ' ClusterId -format '%d ' ProcId "
            f"-format '%s\\n' JobStatus || true"
        )
        res = self.run_remote(cmd, timeout=240)
        if not res.ok:
            return {}
        out: dict[str, str] = {}
        for line in (res.stdout or "").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                cid = parts[0].split(".")[0]
                out[cid] = _external_scheduler_STATUS.get(parts[-1], "I")
        return out

    def external_scheduler_hold_reason(self, cluster_id: str) -> str:
        """Fetch bounded external scheduler hold reason text for classifier evidence."""
        if not cluster_id:
            return ""
        cmd = (
            "timeout 30s scheduler_queue " + shlex.quote(cluster_id) +
            " -hold -af HoldReason HoldReasonCode HoldReasonSubCode RequestMemory 2>/dev/null || true"
        )
        res = self.run_remote(cmd, timeout=45)
        return (res.stdout or "").strip()[:2000] if res.ok else ""

    def poll_sge(self, handles: list[JobHandle]) -> dict[str, str]:
        """Poll SGE job status. Returns {job_id: 'running'|'done'}."""
        ids = [h.cluster_id for h in handles if h.cluster_id and h.scheduler == "sge"]
        if not ids:
            return {}
        ids_str = " ".join(ids)
        cmd = f"qstat -u lhu1"
        # Monitor ticks must stay bounded; CRC checks beyond ~240s should be
        # treated as infrastructure trouble, not model evidence.
        res = self.run_remote(cmd, timeout=240)
        out: dict[str, str] = {}
        if not res.ok:
            return out
        running_ids = set()
        for line in (res.stdout or "").splitlines():
            parts = line.strip().split()
            if parts:
                col0 = parts[0]  # job ID is first column
                if col0 in ids:
                    running_ids.add(col0)
        for jid in ids:
            out[jid] = "running" if jid in running_ids else "done"
        return out

    def check_remote_sentinel(self, handle: JobHandle) -> Optional[str]:
        """Check FINISHED/FAILED/metrics.json on shared filesystem (V3 pattern)."""
        rd = handle.remote_results_dir
        if not rd:
            return None
        # Per-experiment sentinel
        exp_dir = f"{rd}/{handle.experiment_id}" if not rd.endswith(handle.experiment_id) else rd
        cmd = (
            f"if [ -f {shlex.quote(exp_dir + '/FINISHED')} ]; then echo FINISHED; "
            f"elif [ -f {shlex.quote(exp_dir + '/FAILED')} ]; then echo FAILED; "
            f"elif [ -f {shlex.quote(exp_dir + '/metrics.json')} ]; then echo METRICS; "
            f"else echo NONE; fi"
        )
        res = self.run_remote(cmd, timeout=240)
        if not res.ok:
            return None
        token = (res.stdout or "").strip().splitlines()[-1:]
        marker = token[0] if token else ""
        if marker in ("FINISHED", "METRICS"):
            return "completed"
        if marker == "FAILED":
            return "failed"
        return None

    def fetch_remote_metrics(self, handle: JobHandle) -> Optional[dict]:
        """Read metrics.json from shared filesystem."""
        rd = handle.remote_results_dir
        if not rd:
            return None
        exp_dir = f"{rd}/{handle.experiment_id}" if not rd.endswith(handle.experiment_id) else rd
        cmd = f"cat {shlex.quote(exp_dir + '/metrics.json')}"
        res = self.run_remote(cmd, timeout=240)
        if not res.ok or not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def tail_external_scheduler_logs(self, handle: JobHandle, n: int = 30) -> str:
        """Tail external scheduler.out + external scheduler.err for crash detection."""
        rd = handle.remote_results_dir
        if not rd:
            return ""
        exp_dir = f"{rd}/{handle.experiment_id}" if not rd.endswith(handle.experiment_id) else rd
        # Tail files independently. Some remote tail implementations return a
        # non-zero code when one file is empty/missing in a multi-file call,
        # which made crash logs invisible to the monitor.
        files = [exp_dir + "/external scheduler.log", exp_dir + "/external scheduler.out", exp_dir + "/external scheduler.err", exp_dir + "/train.log"]
        cmd = " ; ".join(
            f"echo ---{shlex.quote(f)}---; tail -n {int(n)} {shlex.quote(f)} 2>/dev/null || true"
            for f in files
        )
        res = self.run_remote(cmd, timeout=240)
        return (res.stdout or "").strip()

    def tail_sge_logs(self, handle: JobHandle, n: int = 30) -> str:
        """Tail SGE .out + .err for crash detection."""
        rd = handle.remote_results_dir
        if not rd:
            return ""
        cmd = f"tail -{n} {shlex.quote(f'{rd}/{handle.experiment_id}.out')} {shlex.quote(f'{rd}/{handle.experiment_id}.err')} 2>/dev/null"
        res = self.run_remote(cmd, timeout=DEFAULT_SSH_TIMEOUT)
        return (res.stdout or "").strip()

    def detect_crash(self, log_text: str) -> Optional[str]:
        """Detect crash patterns in log text. Returns crash type or None."""
        patterns = [
            (r"Traceback \(most recent call last\)", "Traceback"),
            (r"\bRuntimeError\b", "RuntimeError"),
            (r"\bTypeError\b", "TypeError"),
            (r"\bCUDA out of memory\b", "OOM"),
            (r"\bOutOfMemoryError\b", "OOM"),
            (r"\bKilled\b", "Killed"),
            (r"\bFileNotFoundError\b", "FileNotFoundError"),
            (r"\bModuleNotFoundError\b", "ImportError"),
            (r"\bImportError\b", "ImportError"),
            (r"\bShapeError\b", "ShapeError"),
            (r"\bValueError\b.*expected", "ValueError"),
            (r"\bSegmentation fault\b", "SegFault"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, log_text):
                return label
        return None

    # â”€â”€ High-level poll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def poll_handles(self, handles: list[JobHandle]) -> list[JobHandle]:
        """Update status for all handles."""
        # Split by scheduler
        external_scheduler_handles = [h for h in handles if h.scheduler == "external scheduler"]
        sge_handles = [h for h in handles if h.scheduler == "sge"]

        # Poll external scheduler
        external_scheduler_statuses = self.poll_external_scheduler(external_scheduler_handles)
        for h in external_scheduler_handles:
            if h.status in ("completed", "failed"):
                continue
            code = external_scheduler_statuses.get(h.cluster_id or "")
            if code:
                logical = _external_scheduler_LOGICAL.get(code, h.status)
                if code == "X":
                    # Evicted â€” check for checkpoint
                    sentinel = self.check_remote_sentinel(h)
                    if sentinel:
                        h.status = sentinel
                    else:
                        h.status = "evicted"
                else:
                    h.status = logical
                    if logical == "held":
                        reason = self.external_scheduler_hold_reason(h.cluster_id or "")
                        if reason:
                            h.log_tail = (h.log_tail + "\n" if h.log_tail else "") + "external_scheduler_HOLD_REASON: " + reason
            else:
                # Not in queue â€” check sentinel
                sentinel = self.check_remote_sentinel(h)
                if sentinel:
                    h.status = sentinel
                else:
                    # The job may have started and exited quickly before the
                    # runner ever observed it as running. external scheduler then removes it
                    # from the queue, no FINISHED/FAILED sentinel is present,
                    # and status can remain "submitted" forever. Inspect the
                    # external scheduler logs immediately for terminal/crash evidence.
                    logs = self.tail_external_scheduler_logs(h, 20)
                    crash = self.detect_crash(logs)
                    if crash or "Job terminated" in logs or "return value 1" in logs:
                        h.status = "failed" if (crash or "return value 1" in logs) else "completed"
                        h.log_tail = logs[-2000:]
                    elif h.status == "running":
                        # Was running, now gone, no sentinel â€” likely crashed
                        h.status = "failed" if crash else "completed"
                        h.log_tail = logs[-2000:]
                    elif h.status == "submitted" and h.submitted_at > 0 and time.time() - h.submitted_at > 600:
                        # Submitted for >10 min, not in queue, no sentinel â€” likely failed
                        h.status = "failed"
                        h.log_tail = logs[-2000:]
                        h.error = "timeout: job disappeared after 10 min without sentinel"

        # Poll SGE
        sge_statuses = self.poll_sge(sge_handles)
        for h in sge_handles:
            if h.status in ("completed", "failed"):
                continue
            status = sge_statuses.get(h.cluster_id or "", "done")
            if status == "running":
                h.status = "running"
            else:
                # Done â€” check for output
                logs = self.tail_sge_logs(h, 20)
                crash = self.detect_crash(logs)
                h.status = "failed" if crash else "completed"
                h.log_tail = logs[-2000:]

        return handles

    # â”€â”€ Batch operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def cancel_external_scheduler(self, handles: list[JobHandle]) -> None:
        for h in handles:
            if h.cluster_id and h.scheduler == "external scheduler":
                self.run_remote(f"timeout 20s scheduler_remove {shlex.quote(h.cluster_id)} || true", timeout=30)

    def cancel_sge(self, handles: list[JobHandle]) -> None:
        for h in handles:
            if h.cluster_id and h.scheduler == "sge":
                self.run_remote(f"qdel {shlex.quote(h.cluster_id)}")

    def submit_analysis_task(
        self,
        task_name: str,
        python_code: str,
        output_file: str,
    ) -> JobHandle:
        """Submit a Python analysis script (inline code) via qsub.

        Writes the script to CRC, then submits as CPU task.
        """
        # Write script to remote
        script_remote = f"{self.remote.analysis_dir(self.campaign_id)}/{task_name}.py"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(python_code)
            tmp_script = Path(f.name)
        try:
            self.run_remote(f"mkdir -p {shlex.quote(self.remote.analysis_dir(self.campaign_id))}")
            self.scp_upload(tmp_script, script_remote)
        finally:
            tmp_script.unlink(missing_ok=True)

        return self.submit_cpu_task(
            task_name=task_name,
            python_script=script_remote,
            working_dir=self.remote.analysis_dir(self.campaign_id),
        )








