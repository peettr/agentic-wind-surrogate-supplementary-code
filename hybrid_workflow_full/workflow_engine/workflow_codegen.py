"""Sequential Workflow Codegen â€” Generate and fix model .py files.

Two modes:
1. Normal mode: generate model code from spec, using V3 models as reference
2. Fix mode: read smoke_fix_plan + error logs, AI patches the broken code

Hard rules:
- No nan_to_num (must use NaN masking)
- Must be nn.Module with forward(x) -> x
- Reflection padding, not zero padding
- Compatible with shared/train.py contract

Called by runner during 'codegen' and 'ai_fix' phases.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "workflow_engine"))
sys.path.insert(0, str(PROJECT_ROOT / "explorer"))

from workflow_common import (
    now_iso, load_state, round_artifact_dir, experiment_id,
)
from schema_guards import (
    input_channels_for_features,
    validate_registered_arch_config,
)

import re as _re

LOGGER = logging.getLogger("hybrid.codegen")
POSTPROCESS_ACTIONS: list[dict] = []


def _record_postprocess(arch_name: str, actions: list[str]) -> None:
    for action in actions:
        POSTPROCESS_ACTIONS.append({"arch_name": arch_name, "action": action})


def _fix_init_sig(code: str, arch_name: str, arch_kwargs: dict) -> str:
    """Patch __init__ signature to include all arch_kwargs params."""
    base_keys = {"in_channels", "out_channels", "n_c", "depth"}
    # Only model-constructor kwargs belong in generated __init__ defaults.  In
    # validation-feedback mode this helper used to receive a whole training cfg,
    # which leaked metadata such as experiment_id=r024_... into the signature as
    # a bare (undefined) variable.  Keep real architecture knobs and drop runner
    # / training fields defensively.
    metadata_keys = {
        "experiment_id", "strategy", "arch_name", "loss_name", "loss_kwargs",
        "lr", "batch_size", "input_features", "epochs", "seed", "use_ema",
        "ema_decay", "augmentation", "data_dir", "results_dir",
        "split_manifest_path", "script_path", "_config_repair_patch",
        "_schema_repair_patch", "_repair_base_id", "_repair_source_run_id",
    }
    safe_kwargs = {
        str(k): v
        for k, v in (arch_kwargs or {}).items()
        if k not in metadata_keys and str(k).isidentifier()
    }
    required_extra = {k: v for k, v in safe_kwargs.items() if k not in base_keys}
    if not required_extra:
        return code
    pattern = rf"(class {_re.escape(arch_name)}\b[^:]*:.*?\n    def __init__\(self,\s*)([^)]+)\)"
    m = _re.search(pattern, code, _re.DOTALL)
    if not m:
        return code
    existing = set()
    for p in m.group(2).split(","):
        p = p.strip().split("=")[0].strip()
        if p:
            existing.add(p)
    missing = {k: v for k, v in required_extra.items() if k not in existing}
    if not missing:
        return code
    new_params = ", ".join(f"{k}={repr(v)}" for k, v in missing.items())
    old_sig = m.group(0)
    new_sig = m.group(1) + m.group(2).strip() + ", " + new_params + ")"
    code = code.replace(old_sig, new_sig)
    LOGGER.info("Patched __init__ for %s: added %s", arch_name, list(missing.keys()))
    return code

CLAUDE_BIN_FULL = shutil.which("claude") or shutil.which("claude.exe") or "claude"
CODEX_BIN_FULL = shutil.which("codex") or "codex"
CODEX_MODEL = os.environ.get("hybrid_CODEGEN_CODEX_MODEL", "gpt-5.5")
CLAUDE_FIX_MODEL = os.environ.get("hybrid_FIX_CLAUDE_MODEL", "claude-opus-4-7")

MODELS_DIR = PROJECT_ROOT / "shared" / "models"
GENERATED_DIR = PROJECT_ROOT / "models" / "generated"


def _env_enabled(name: str, default: str = "1") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in {"0", "false", "no", "off"}


def _repair_trace_dir() -> Path:
    trace_dir = GENERATED_DIR / "_repair_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _write_repair_trace(arch_name: str, provider: str, attempt: int, raw: str = "", validation_error: str = "") -> None:
    """Persist repair CLI output and validation context without overwriting."""
    try:
        import time
        safe_arch = re.sub(r"[^A-Za-z0-9_.-]+", "_", arch_name)
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{os.getpid()}_{len(list(_repair_trace_dir().glob(f'{safe_arch}_{provider}_{ts}_*.json')))}"
        path = _repair_trace_dir() / f"{safe_arch}_{provider}_{ts}_{suffix}.json"
        payload = {
            "arch_name": arch_name,
            "provider": provider,
            "attempt": attempt,
            "model": CLAUDE_FIX_MODEL if provider == "claude" else CODEX_MODEL,
            "validation_error": validation_error,
            "raw": raw,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        LOGGER.warning("failed to write repair trace for %s/%s: %s", arch_name, provider, e)


def _registered_arch_names() -> set[str]:
    """Read shared/models registry keys without importing torch-heavy modules."""
    init_path = MODELS_DIR / "__init__.py"
    if not init_path.exists():
        return set()
    text = init_path.read_text(encoding="utf-8")
    return set(re.findall(r'REGISTRY\.register\("([^"]+)"', text))


def _make_standalone_model_code(code: str) -> str:
    """Make generated/reference model files loadable via spec_from_file_location.

    shared/models are package modules and often use ``from .base import
    BaseSurrogate``.  Generated models are shipped to each run directory and
    loaded as standalone files by shared/train.py, so relative imports are not
    valid there.
    """
    code = re.sub(r"^\s*from\s+\.base\s+import\s+BaseSurrogate\s*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from\s+\.[\w_]+\s+import\s+.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"\(\s*BaseSurrogate\s*\)", "(nn.Module)", code)
    return code


def _ensure_arch_class_export(code: str, arch_name: str) -> str:
    """Ensure standalone model files expose train.py's exact cfg.arch_name.

    Reference model filenames and internal class names often differ from the
    registry/config key, for example ``unet_v3.py`` defines ``UNet`` and
    ``unet_sdf_7level.py`` defines ``UNetSDF``.  The training loader resolves
    dynamic script_path files by exact attribute lookup on ``cfg.arch_name``.
    Without this wrapper, local codegen validation may pass by finding any
    nn.Module subclass, while remote training fails with
    ``No torch.nn.Module class named '<arch_name>' in model.py``.
    """
    try:
        import ast
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if any(isinstance(node, ast.ClassDef) and node.name == arch_name for node in tree.body):
        return code

    helper_names = {
        "DoubleConv", "ConvBlock", "Down", "Up", "ResBlock", "Block",
        "Encoder", "Decoder", "AttentionBlock", "SpectralBlock",
    }
    candidates: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name in helper_names:
            continue
        method_names = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
        if "forward" in method_names and "__init__" in method_names:
            candidates.append(node.name)
    if not candidates:
        return code

    base_cls = candidates[-1]
    wrapper = f'''

# Hybrid standalone export shim.  Keep this exact cfg.arch_name class for train.py.
class {arch_name}({base_cls}):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, **kwargs):
        import inspect
        sig = inspect.signature({base_cls}.__init__)
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
    return code.rstrip() + wrapper


def _copy_missing_helpers(fixed_code: str, reference_code: str) -> str:
    """Copy referenced top-level helper defs/classes from reference code.

    Codex often follows the old prompt too literally and returns only the main
    class while still referencing helpers such as _gn, ResBlock, or
    SimpleSSMBlock.  Preserve any referenced helper definitions from the trusted
    reference file before validation.
    """
    import ast
    try:
        fixed_tree = ast.parse(fixed_code)
        ref_tree = ast.parse(reference_code)
    except SyntaxError:
        return fixed_code

    fixed_defs = {
        node.name for node in fixed_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    segments: list[str] = []
    for node in ref_tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name in fixed_defs:
            continue
        if not re.search(rf"\b{re.escape(node.name)}\b", fixed_code):
            continue
        seg = ast.get_source_segment(reference_code, node)
        if seg:
            segments.append(seg)
            fixed_defs.add(node.name)

    if not segments:
        return fixed_code

    lines = fixed_code.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
            insert_at = i + 1
            continue
        break
    return "\n".join(lines[:insert_at]) + "\n\n" + "\n\n".join(segments) + "\n\n" + "\n".join(lines[insert_at:])


def normalize_future_imports(code: str) -> tuple[str, list[str]]:
    """Hoist/dedupe __future__ imports after an optional module docstring.

    Python requires future imports before regular imports/code. AI repairs and
    helper injection can accidentally leave them mid-file, so normalize before
    compile/write in both generate and repair paths.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    future_lines: list[str] = []
    remove_lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            names = ", ".join(alias.name for alias in node.names)
            line = f"from __future__ import {names}"
            if line not in future_lines:
                future_lines.append(line)
            for lineno in range(getattr(node, "lineno", 1), getattr(node, "end_lineno", getattr(node, "lineno", 1)) + 1):
                remove_lines.add(lineno)

    if not future_lines:
        return code, []

    lines = code.splitlines()
    kept = [line for idx, line in enumerate(lines, start=1) if idx not in remove_lines]
    insert_at = 0
    try:
        kept_tree = ast.parse("\n".join(kept) + "\n")
        if kept_tree.body and isinstance(kept_tree.body[0], ast.Expr) and isinstance(getattr(kept_tree.body[0], "value", None), ast.Constant) and isinstance(kept_tree.body[0].value.value, str):
            insert_at = getattr(kept_tree.body[0], "end_lineno", 1)
            while insert_at < len(kept) and not kept[insert_at].strip():
                insert_at += 1
    except SyntaxError:
        insert_at = 0

    new_lines = kept[:insert_at] + future_lines + [""] + kept[insert_at:]
    normalized = "\n".join(new_lines).rstrip() + "\n"
    actions = []
    if normalized != code:
        actions.append("normalize_future_imports:hoist_dedupe")
    return normalized, actions


def _inject_builtin_helpers(code: str) -> str:
    """Inject tiny safe helpers that codegen commonly references.

    These helpers are deliberately simple and self-contained so generated
    standalone files do not fail validation because Codex referenced a helper
    name without defining it.
    """
    helpers: list[str] = []
    if re.search(r"\b_mask_nans\s*\(", code) and not re.search(r"def\s+_mask_nans\s*\(", code):
        helpers.append(
            "def _mask_nans(x: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))\n"
        )

    if re.search(r"\b_valid_groups\s*\(", code) and not re.search(r"def\s+_valid_groups\s*\(", code):
        helpers.append(
            "def _valid_groups(channels: int, max_groups: int = 32) -> int:\n"
            "    # Return the largest GroupNorm group count <= max_groups that divides channels.\n"
            "    for groups in (32, 16, 8, 4, 2, 1):\n"
            "        if groups <= max_groups and channels % groups == 0:\n"
            "            return groups\n"
            "    return 1\n"
        )

    if not helpers:
        return code

    lines = code.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
            insert_at = i + 1
            continue
        break
    return "\n".join(lines[:insert_at]) + "\n\n" + "\n\n".join(helpers) + "\n" + "\n".join(lines[insert_at:])


def run_cmd(cmd: list[str], timeout: int = 900, stdin_text: str | None = None) -> tuple[bool, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       timeout=timeout, input=stdin_text)
    # Only use stdout for code extraction; stderr is logged but not mixed in
    text = (r.stdout or "").strip()
    if r.returncode != 0 and r.stderr:
        LOGGER.warning("cmd stderr: %s", r.stderr.strip()[:200])
    return r.returncode == 0, text


# -- Reference file management ------------------------------------------

def find_reference_model(arch_name: str) -> Path | None:
    """Find a shared model .py as reference for codegen/fix.

    Registry keys and filenames are not always identical, e.g.
    ``perceiver_io`` is implemented in ``shared/models/perceiver.py``.  Use a
    conservative two-way fuzzy match after exact lookup so repair has a real
    reference file instead of failing with "no existing code".
    """
    exact = MODELS_DIR / f"{arch_name}.py"
    if exact.exists():
        return exact
    norm = arch_name.replace("_", "").lower()
    aliases = {
        "perceiverio": "perceiver",
    }
    alias = aliases.get(norm)
    if alias:
        p = MODELS_DIR / f"{alias}.py"
        if p.exists():
            return p
    for f in MODELS_DIR.glob("*.py"):
        stem = f.stem.replace("_", "").lower()
        if norm in stem or stem in norm:
            return f
    return None


def load_reference_code(arch_name: str) -> str | None:
    """Read reference model code from V3 library."""
    ref = find_reference_model(arch_name)
    if ref:
        return ref.read_text(encoding="utf-8", errors="replace")
    return None


# -- Normal mode: generate from spec ------------------------------------

def build_generation_prompt(cfg: dict, reference_code: str | None = None) -> str:
    """Build prompt with skeleton template (Codex review recommendation #2)."""
    arch_name = cfg.get("arch_name")
    arch_kwargs = cfg.get("arch_kwargs", {})
    base_keys = {"in_channels", "out_channels", "n_c", "depth"}
    extra = {k: v for k, v in arch_kwargs.items() if k not in base_keys}

    # Build exact __init__ signature
    sig_parts = ["in_channels=1", "out_channels=1", f"n_c={cfg.get('n_c', 16)}", f"depth={cfg.get('depth', 7)}"]
    for k, v in extra.items():
        sig_parts.append(f"{k}={v}")
    init_sig = ", ".join(sig_parts)

    # Build skeleton stub
    modes_hint = ""
    if "modes" in extra:
        modes_hint = f"- modes={extra['modes']}: use for FFT truncation (e.g., x_ft[:,:,:modes,:modes]), NOT for weight tensor size\n"
    skeleton = (
        f"import torch\n"
        f"import torch.nn as nn\n"
        f"import torch.nn.functional as F\n\n"
        f"class {arch_name}(nn.Module):\n"
        f"    def __init__(self, {init_sig}):\n"
        f"        super().__init__()\n"
        f"        # TODO: build encoder, decoder, bottleneck\n"
        f"        # CRITICAL: total params must be under 50M.\n"
        f"        # Use n_c as base channel count. Max channels = n_c * 8.\n"
        f"        # If using spectral/Fourier: truncate FFT, do NOT create large weight tensors.\n"
        f"        pass\n\n"
        f"    def forward(self, x):\n"
        f"        # TODO: implement forward pass\n"
        f"        # Output must be same size as input (B, 1, 640, 640)\n"
        f"        return x\n"
    )

    # Extra params description
    extra_desc = ""
    if "dilation" in extra:
        extra_desc += f"- dilation={extra['dilation']}: apply to ALL conv layers in encoder and decoder\n"
    if "modes" in extra:
        extra_desc += f"- modes={extra['modes']}: FFT truncation only. Use x_ft[:,:,:modes,:modes] to truncate frequencies. Do NOT create any weight tensors for spectral layers. Just truncate and transform.\n"
    for k, v in extra.items():
        if k not in ("dilation", "modes"):
            extra_desc += f"- {k}={v}\n"

    prompt = (
        f"Complete this PyTorch model skeleton for wind pressure prediction.\n\n"
        f"Task: 640x640 height map -> 640x640 pressure field.\n\n"
        f"Fill in the skeleton below. Return raw Python only (no markdown fences, no prose):\n\n"
        f"{skeleton}\n"
        f"Rules:\n"
        f"- Use nn.ReflectionPad2d for all padding (NOT zero padding)\n"
        f"- NaN masking: x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x); after forward, restore NaN: output[~valid] = NaN\n"
        f"- Import only torch, torch.nn, torch.nn.functional\n"
        f"- Channel schedule: n_c, 2*n_c, 4*n_c, ... up to depth levels\n"
        f"- Size-preserving: input and output must both be (B, 1, 640, 640)\n"
    )
    if extra_desc:
        prompt += f"\nExtra parameter details:\n{extra_desc}\n"
    return prompt



def generate_model_code(arch_name: str, cfg: dict, round_num: int = 0) -> tuple[bool, str]:
    """Generate model .py using Codex CLI with skeleton template."""
    import ast
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = GENERATED_DIR / f"{arch_name}.py"

    # Skip if already exists AND validated
    if model_path.exists():
        val_ok, _ = validate_model_code_static(arch_name, cfg)
        if val_ok:
            return True, "exists"
        LOGGER.warning("Regenerating broken %s", arch_name)
        model_path.unlink(missing_ok=True)

    # No reference code Ã¢â‚¬â€ it triggers Codex agentic loop (timeout)
    prompt = build_generation_prompt(cfg, reference_code=None)

    for attempt in range(3):
        ok, out = run_cmd(
            [CODEX_BIN_FULL, "exec", "--model", CODEX_MODEL, "--skip-git-repo-check",
             "--cd", "/tmp", "--ephemeral"],
            timeout=900, stdin_text=prompt,
        )

        if not ok or not out:
            if attempt < 2:
                LOGGER.warning("Attempt %d no output for %s, retrying", attempt, arch_name)
            continue

        code = extract_code_block(out)
        if not code:
            if attempt < 2 and len(out) > 50:
                LOGGER.warning("Attempt %d no code block for %s, retrying", attempt, arch_name)
                continue
            return False, f"no code block in output ({len(out)} chars)"

        # AST validation (Codex recommendation #3)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if attempt < 2:
                LOGGER.warning("Attempt %d syntax error for %s: %s, retrying", attempt, arch_name, e)
                continue
            return False, f"syntax error: {e}"

        # Check: at least one class with correct name (allow helper classes)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        matching = [c for c in classes if c.name == arch_name]
        if not matching:
            camel = ''.join(w.capitalize() for w in arch_name.split('_'))
            matching = [c for c in classes if c.name == camel]
            if matching:
                code = code.replace(f'class {camel}(', f'class {arch_name}(', 1)
                LOGGER.info("Renamed %s -> %s", camel, arch_name)
            else:
                if attempt < 2:
                    found_names = [c.name for c in classes]
                    LOGGER.warning("Attempt %d: no class %s (found %s), retrying", attempt, arch_name, found_names)
                    continue
                return False, f"no class {arch_name}"

        # Verify __init__ has all required extra params (hard gate)
        arch_kwargs = cfg.get("arch_kwargs", {})
        base_keys = {"in_channels", "out_channels", "n_c", "depth"}
        required_extra = {k: v for k, v in arch_kwargs.items() if k not in base_keys}
        if required_extra:
            # Parse __init__ from the generated code
            init_match = re.search(r'def __init__\(self,([^)]*)\)', code)
            if init_match:
                init_params = set()
                for p in init_match.group(1).split(','):
                    p = p.strip().split('=')[0].strip()
                    if p and p != 'self':
                        init_params.add(p)
            else:
                init_params = set()

            missing = [k for k in required_extra if k not in init_params]
            if missing:
                if attempt < 2:
                    LOGGER.warning("Attempt %d: missing __init__ params %s for %s, retrying", attempt, missing, arch_name)
                    # Build a more explicit retry prompt
                    sig_parts = ["in_channels=1", "out_channels=1", f"n_c={cfg.get('n_c', 16)}", f"depth={cfg.get('depth', 7)}"]
                    for k, v in required_extra.items():
                        sig_parts.append(f"{k}={v}")
                    full_sig = ", ".join(sig_parts)
                    prompt = (
                        f"Complete this EXACT skeleton. Do NOT change the __init__ signature.\n\n"
                        f"import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
                        f"class {arch_name}(nn.Module):\n"
                        f"    def __init__(self, {full_sig}):\n"
                        f"        super().__init__()\n"
                        f"        # CRITICAL: total params must be under 50M.\n"
                        f"        # fill in here\n"
                        f"    def forward(self, x):\n"
                        f"        # fill in here\n"
                        f"        return x\n\n"
                        f"Return raw Python only. Use nn.ReflectionPad2d. NaN masking with torch.isnan."
                    )
                    continue
                return False, f"missing __init__ params: {missing}"

# Check __init__ has all required params (Codex recommendation #2)
        arch_kwargs = cfg.get("arch_kwargs", {})
        base_keys = {"in_channels", "out_channels", "n_c", "depth"}
        required_extra = {k: v for k, v in arch_kwargs.items() if k not in base_keys}
        if required_extra:
            code = _fix_init_sig(code, arch_name, arch_kwargs)
            # Verify fix worked
            for param in required_extra:
                if param not in code.split("def __init__")[1].split(")")[0]:
                    if attempt < 2:
                        LOGGER.warning("Attempt %d: missing param %s, retrying", attempt, param)
                        break
                    return False, f"missing __init__ param: {param}"
            else:
                # All params present, continue to write
                pass
        else:
            # No extra params to check
            pass

        # Check for required patterns
        if "nn.ReflectionPad2d" not in code and "ReflectionPad2d" not in code:
            if attempt < 2:
                LOGGER.warning("Attempt %d: no ReflectionPad2d for %s, retrying", attempt, arch_name)
                continue
            return False, "missing nn.ReflectionPad2d"

        if "nan_to_num" in code:
            if attempt < 2:
                LOGGER.warning("Attempt %d: uses nan_to_num for %s, retrying", attempt, arch_name)
                continue
            return False, "uses nan_to_num (should use NaN masking)"

        # Force-fix __init__ signature (Codex sometimes ignores skeleton signature)
        code = _make_standalone_model_code(code)
        code = _ensure_arch_class_export(code, arch_name)
        code = _inject_builtin_helpers(code)
        code = _fix_init_sig(code, arch_name, cfg.get("arch_kwargs", {}))
        code, actions = normalize_future_imports(code)
        _record_postprocess(arch_name, actions)
        model_path.write_text(code, encoding="utf-8")
        return True, f"generated {len(code)} chars"

    return False, "all attempts failed"


# ---- Fix mode: patch broken code ----

def build_fix_prompt(arch_name: str, error_log: str, diagnosis: str,
                     fix_description: str, current_code: str,
                     validation_error: str = "") -> str:
    validation_block = ""
    if validation_error:
        validation_block = (
            "Previous generated fix failed validation. Fix this validation failure "
            "while preserving the original repair intent.\n"
            f"Validation error:\n```\n{validation_error[:3000]}\n```\n\n"
        )
    return (
        f"You are fixing a broken neural network model for wind pressure prediction.\n\n"
        f"Model: {arch_name}\n\n"
        f"Diagnosis: {diagnosis}\n"
        f"Fix needed: {fix_description}\n\n"
        f"Error log (last lines):\n```\n{error_log[:2000]}\n```\n\n"
        f"{validation_block}"
        f"Current broken code:\n```python\n{current_code[:16000]}\n```\n\n"
        f"Apply the MINIMAL fix. Output ONLY the corrected FULL Python file, not just the class.\n"
        f"The file must be self-contained: include imports and every helper function/class referenced by the model (for example _gn, ResBlock, SpectralBottleneck, SimpleSSMBlock).\n"
        f"Do not place 'from __future__ import ...' in the middle of the file; if used, it must appear only at the top after the optional module docstring.\n"
        f"The primary model class MUST keep its original name: '{arch_name}' (snake_case, NOT CamelCase).\n"
        f"Requirements:\n"
        f"1. nn.Module subclass with __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7)\n"
        f"2. forward(self, x) -> x\n"
        f"3. Use reflection padding, not zero padding\n"
        f"4. No nan_to_num - use NaN masking\n"
    )


def _is_claude_fix_enabled() -> bool:
    return _env_enabled("hybrid_FIX_CLAUDE_ENABLED", "1")


def _is_code_repair_error(message: str) -> bool:
    """Return True for validation failures likely caused by model code bugs."""
    msg = (message or "").lower()
    if not msg:
        return False
    code_bug_terms = (
        "shape", "size mismatch", "channel", "channels", "mask", "masked",
        "groupnorm", "group norm", "import", "module", "helper", "not defined",
        "nameerror", "attributeerror", "class", "forward", "instantiation",
        "conv2d", "input_channel_mismatch", "missing import", "missing nn.module",
        "missing forward", "syntax", "relative import", "basesurrogate",
    )
    non_code_terms = (
        "oom", "out of memory", "cuda error", "time limit", "timeout",
        "external_scheduler", "held", "preempt", "disk quota", "no space", "permission denied",
    )
    return any(t in msg for t in code_bug_terms) and not any(t in msg for t in non_code_terms)


def _should_try_claude_fix(fix_info: dict, validation_error: str, attempt: int) -> bool:
    """Gate the one-shot Claude fallback after a failed Codex code repair."""
    if not _is_claude_fix_enabled():
        return False
    if str(fix_info.get("fix_type", "code")).lower() != "code":
        return False
    message = "\n".join(str(fix_info.get(k, "") or "") for k in ("validation_error", "error_msg", "error_log", "diagnosis", "fix_description"))
    message = f"{message}\n{validation_error or ''}"
    return _is_code_repair_error(message) or attempt >= 1


def build_claude_fix_prompt(arch_name: str, current_code: str, validation_error: str, fix_info: dict) -> str:
    return (
        "You are doing a targeted one-shot repair of a single generated Python model file.\n\n"
        f"Target file/class: {arch_name}.py / class {arch_name}\n"
        "Constraints:\n"
        "- Fix ONLY this architecture file. Do not modify configs, data, training, eval, templates, jobs, or other files.\n"
        f"- Keep the primary class name exactly `{arch_name}`.\n"
        "- Preserve the input/output contract: nn.Module, __init__(in_channels=1, out_channels=1, n_c=16, depth=7, ...), forward(self, x) -> tensor with shape (B, 1, H, W).\n"
        "- Keep the file standalone: include all imports and every helper function/class it references.\n"
        "- Use reflection padding, not zero padding. Do not use torch.nan_to_num; use explicit NaN masking if needed.\n"
        "- Prefer directly outputting the complete corrected Python file, and output no prose. If you edit the file directly, still print the complete final Python file to stdout.\n\n"
        f"Diagnosis: {fix_info.get('diagnosis', 'unknown')}\n"
        f"Requested fix: {fix_info.get('fix_description', 'fix the validation error')}\n"
        f"Validation/error message to fix:\n```\n{validation_error[:4000]}\n```\n\n"
        f"Current Python file:\n```python\n{current_code[:24000]}\n```\n"
    )


def _validation_cfg_from_fix_info(fix_info: dict) -> dict:
    cfg: dict = {}
    arch_kwargs = fix_info.get("arch_kwargs") or {}
    if isinstance(arch_kwargs, dict):
        cfg.update(arch_kwargs)
        cfg["arch_kwargs"] = arch_kwargs
    for key in ("input_features", "n_c", "depth"):
        if key in fix_info:
            cfg[key] = fix_info[key]
    return cfg


def _prepare_repaired_code(arch_name: str, code: str, current_code: str, arch_kwargs: dict | None = None) -> str:
    if 'import torch.nn as nn' not in code and 'import torch.nn' not in code:
        code = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n' + code
    elif 'import torch.nn as nn' not in code and 'import torch.nn' in code:
        code = code.replace('import torch.nn', 'import torch\nimport torch.nn as nn', 1)
    if "import torch" not in code:
        code = "import torch\nimport torch.nn as nn\n\n" + code
    code = _make_standalone_model_code(code)
    code = _copy_missing_helpers(code, current_code)
    code = _ensure_arch_class_export(code, arch_name)
    code = _inject_builtin_helpers(code)
    code = _fix_init_sig(code, arch_name, arch_kwargs or {})
    code, actions = normalize_future_imports(code)
    _record_postprocess(arch_name, actions)
    return code


def _validate_written_repair(arch_name: str, cfg: dict) -> tuple[bool, str]:
    static_ok, static_msg = validate_model_code_static(arch_name, cfg)
    if not static_ok:
        return False, f"static validation: {static_msg}"
    return validate_model_code(arch_name, cfg)


def fix_model_code(arch_name: str, fix_info: dict) -> tuple[bool, str]:
    """Fix broken model code using Codex CLI.

    This path is for code repairs only.  Config/resource fixes, e.g. OOM with
    smaller batch or H100 retry, are handled by controller/submit logic and must
    not rewrite model files.
    """
    fix_type = str(fix_info.get("fix_type", "code")).lower()
    if fix_type != "code":
        return False, f"non-code fix_type={fix_info.get('fix_type')} should not run codegen"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = GENERATED_DIR / f"{arch_name}.py"
    # Prefer the trusted shared reference as the repair base.  Previous failed
    # repair attempts may have left a generated file that is missing helpers or
    # otherwise invalid; repeatedly repairing that poisoned file compounds the
    # damage.
    validation_error = str(fix_info.get("validation_error", "") or "")
    ref_code = load_reference_code(arch_name)
    # For validation-feedback attempts, repair the failed generated file so the
    # validation error is part of the next context. For the first attempt,
    # prefer the trusted shared reference to avoid compounding stale bad fixes.
    if validation_error and model_path.exists():
        current_code = model_path.read_text(encoding="utf-8")
    elif ref_code:
        current_code = _make_standalone_model_code(ref_code)
        current_code = _ensure_arch_class_export(current_code, arch_name)
        current_code, actions = normalize_future_imports(current_code)
        _record_postprocess(arch_name, actions)
        model_path.write_text(current_code, encoding="utf-8")
    elif model_path.exists():
        current_code = model_path.read_text(encoding="utf-8")
    else:
        return False, f"no existing code for {arch_name}"
    prompt = build_fix_prompt(
        arch_name=arch_name,
        error_log=fix_info.get("error_log", ""),
        diagnosis=fix_info.get("diagnosis", "unknown"),
        fix_description=fix_info.get("fix_description", "fix the error"),
        current_code=current_code,
        validation_error=validation_error,
    )

    repair_attempt = int(fix_info.get("attempt", 1 if validation_error else 0) or 0)
    validation_cfg = _validation_cfg_from_fix_info(fix_info)

    ok, out = run_cmd(
        [CODEX_BIN_FULL, "exec", "--model", CODEX_MODEL, "--skip-git-repo-check",
         "--cd", "/tmp", "--ephemeral", "--ignore-rules"],
        timeout=900, stdin_text=prompt,
    )
    _write_repair_trace(arch_name, "codex", repair_attempt, out or "", validation_error)

    codex_error = ""
    if ok and out:
        code = extract_code_block(out)
        if code:
            code = _prepare_repaired_code(arch_name, code, current_code, fix_info.get("arch_kwargs", {}))
            # Validate syntax
            try:
                compile(code, f"<{arch_name}.py>", "exec")
                model_path.write_text(code, encoding="utf-8")
                val_ok, val_msg = _validate_written_repair(arch_name, validation_cfg)
                if val_ok:
                    return True, f"fixed {len(code)} chars"
                codex_error = val_msg
            except SyntaxError as e:
                codex_error = f"fix syntax error: {e}"
        else:
            codex_error = f"no code block in Codex output ({len(out)} chars)"
    else:
        codex_error = (out or "no output")[:1000]

    if codex_error:
        _write_repair_trace(arch_name, "codex_validation", repair_attempt, out or "", codex_error)

    if _should_try_claude_fix(fix_info, codex_error or validation_error, repair_attempt):
        claude_prompt = build_claude_fix_prompt(
            arch_name=arch_name,
            current_code=model_path.read_text(encoding="utf-8") if model_path.exists() else current_code,
            validation_error=codex_error or validation_error,
            fix_info=fix_info,
        )
        claude_ok, claude_out = run_cmd(
            [CLAUDE_BIN_FULL, "--model", CLAUDE_FIX_MODEL, "--permission-mode", "bypassPermissions", "--print"],
            timeout=900, stdin_text=claude_prompt,
        )
        _write_repair_trace(arch_name, "claude", repair_attempt, claude_out or "", codex_error or validation_error)
        if claude_ok and claude_out:
            claude_code = extract_code_block(claude_out)
            if claude_code:
                claude_code = _prepare_repaired_code(arch_name, claude_code, current_code, fix_info.get("arch_kwargs", {}))
                try:
                    compile(claude_code, f"<{arch_name}.py>", "exec")
                    model_path.write_text(claude_code, encoding="utf-8")
                    val_ok, val_msg = _validate_written_repair(arch_name, validation_cfg)
                    if val_ok:
                        return True, f"fixed by claude fallback {len(claude_code)} chars after Codex failure: {codex_error[:180]}"
                    return False, f"codex failed: {codex_error[:240]}; claude validation failed: {val_msg[:240]}"
                except SyntaxError as e:
                    return False, f"codex failed: {codex_error[:240]}; claude syntax error: {e}"
            return False, f"codex failed: {codex_error[:240]}; no code block in Claude output ({len(claude_out)} chars)"
        return False, f"codex failed: {codex_error[:240]}; claude failed: {(claude_out or 'no output')[:240]}"

    return False, codex_error[:300]


# -- Validation ---------------------------------------------------------

def validate_model_code(arch_name: str, cfg: dict | None = None) -> tuple[bool, str]:
    """Validate generated model code."""
    model_path = GENERATED_DIR / f"{arch_name}.py"
    if not model_path.exists():
        return False, "file not found"

    code = model_path.read_text(encoding="utf-8")
    issues = []

    # Static checks
    if "nn.Module" not in code:
        issues.append("missing nn.Module")
    if "def forward" not in code:
        issues.append("missing forward method")
    if "nan_to_num" in code:
        issues.append("uses nan_to_num (must mask NaN)")
    if "ZeroPad2d" in code:
        issues.append("uses zero padding (must use ReflectionPad2d)")
    if re.search(r"^\s*(from\s+\.|import\s+\.)", code, flags=re.MULTILINE):
        issues.append("uses relative import (standalone script_path load will fail)")
    if "BaseSurrogate" in code:
        issues.append("uses BaseSurrogate (generated files must subclass nn.Module directly)")

    features = (cfg or {}).get("input_features", "height")
    expected_in = input_channels_for_features(features)
    if expected_in is None:
        issues.append(f"unknown input_features={features!r}")

    # Conservative AST/static channel gate: reject explicit Conv2d(1, ...)
    # only for multi-channel feature contracts. Variable in_channels patterns
    # are allowed and then verified by dynamic forward.
    if expected_in and expected_in > 1:
        try:
            import ast
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_conv2d = (
                    (isinstance(func, ast.Attribute) and func.attr == "Conv2d")
                    or (isinstance(func, ast.Name) and func.id == "Conv2d")
                )
                if not is_conv2d:
                    continue
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 1:
                    issues.append("INPUT_CHANNEL_MISMATCH: hardcoded nn.Conv2d(1, ...) incompatible with multi-channel input_features")
                    break
                for kw in node.keywords:
                    if kw.arg == "in_channels" and isinstance(kw.value, ast.Constant) and kw.value.value == 1:
                        issues.append("INPUT_CHANNEL_MISMATCH: hardcoded nn.Conv2d(in_channels=1, ...) incompatible with multi-channel input_features")
                        break
        except SyntaxError as e:
            issues.append(f"syntax error: {e}")

    # Dynamic check
    n_c = (cfg or {}).get("n_c", 16)
    depth = (cfg or {}).get("depth", 7)
    try:
        import torch
        spec = importlib.util.spec_from_file_location(arch_name, model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model_cls = getattr(mod, arch_name, None)
        if not (isinstance(model_cls, type) and issubclass(model_cls, torch.nn.Module)):
            issues.append(f"class {arch_name} not found or not nn.Module")
            model_cls = None
        if model_cls is not None and expected_in is not None:
            try:
                model = model_cls(in_channels=expected_in, out_channels=1, n_c=n_c, depth=depth)
                # Validate at a deterministic depth/window-compatible scale.
                min_side = max(128, 4 * (2 ** max(0, int(depth) - 1)))
                ws_matches = [int(m) for m in re.findall(r"window_size\s*=\s*(\d+)", code)]
                if ws_matches:
                    min_side = max(min_side, max(ws_matches) * (2 ** max(0, int(depth) - 1)))
                side = min(640, min_side)
                x = torch.randn(1, expected_in, side, side)
                y = model(x)
                if y.shape != (1, 1, side, side):
                    issues.append(f"output shape mismatch: {y.shape}")
            except Exception as e:
                issues.append(f"instantiation/forward error: {e}")
    except Exception as e:
        issues.append(f"import error: {e}")

    if issues:
        return False, "; ".join(issues)
    return True, "ok"


def validate_model_code_static(arch_name: str, cfg: dict | None = None) -> tuple[bool, str]:
    """Static-only validation (no torch import, for skip-check)."""
    import ast as _ast
    model_path = GENERATED_DIR / f"{arch_name}.py"
    if not model_path.exists():
        return False, "file not found"
    code = model_path.read_text(encoding="utf-8")
    if "nn.Module" not in code:
        return False, "missing nn.Module"
    if "def forward" not in code:
        return False, "missing forward"
    if "nn." in code and not re.search(r"(^|\n)\s*import\s+torch\.nn\s+as\s+nn\b", code):
        return False, "missing import torch.nn as nn"
    if "torch." in code and not re.search(r"(^|\n)\s*import\s+torch\b", code):
        return False, "missing import torch"
    if "F." in code and not re.search(r"(^|\n)\s*import\s+torch\.nn\.functional\s+as\s+F\b", code):
        return False, "missing import torch.nn.functional as F"
    if "nan_to_num" in code:
        return False, "nan_to_num"
    if re.search(r"^\s*(from\s+\.|import\s+\.)", code, flags=re.MULTILINE):
        return False, "relative import"
    if "BaseSurrogate" in code:
        return False, "BaseSurrogate dependency"
    if "from ." in code or "import ." in code:
        return False, "relative import not allowed for generated single-file model"
    try:
        tree = _ast.parse(code)
    except _ast.SyntaxError:
        return False, "syntax error"
    class_node = None
    for cls in tree.body:
        if isinstance(cls, _ast.ClassDef) and cls.name == arch_name:
            class_node = cls
            break
    if class_node is None:
        return False, f"class {arch_name} not found"

    # Check __init__ signature has all required extra params
    if cfg:
        arch_kwargs = cfg.get("arch_kwargs", {})
        base_keys = {"in_channels", "out_channels", "n_c", "depth"}
        required_extra = {k for k in arch_kwargs if k not in base_keys}
        if required_extra:
            for node in class_node.body:
                if isinstance(node, _ast.FunctionDef) and node.name == "__init__":
                    params = {a.arg for a in node.args.args if a.arg != "self"}
                    has_kwargs = node.args.kwarg is not None
                    missing = required_extra - params
                    if missing and not has_kwargs:
                        return False, f"missing params: {missing}"
                    break
    return True, "ok"


# -- Code extraction ----------------------------------------------------

def extract_code_block(text: str) -> str | None:
    """Extract Python code from AI output.
    Supports: fenced (```python), bare (```), or raw Python (no fences)."""
    text = text.strip()
    # Try fenced code block
    m = re.search(r'```python\n([\s\S]*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\n([\s\S]*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Raw Python: if text starts with 'import' or 'class', treat as code
    if text.startswith(('import ', 'class ', 'from ')):
        return text
    # Try to find class definition anywhere
    if 'class ' in text and 'def forward' in text:
        start = text.find('class ')
        return text[start:].strip()
    return None


# -- Main ---------------------------------------------------------------

def main() -> None:
    campaign_dir = Path(os.environ.get("HYBRID_CAMPAIGN_DIR", "."))
    state = load_state(campaign_dir)
    round_num = state.get("round_num", 0)
    art_dir = round_artifact_dir(campaign_dir, round_num)

    phase = state.get("phase")
    is_fix_mode = state.get("fix_mode", False) or (phase == "ai_fix")

    # Load proposals
    proposals = state.get("proposals", [])
    if not proposals:
        proposals_path = art_dir / "proposals.json"
        if proposals_path.exists():
            proposals = json.loads(proposals_path.read_text(encoding="utf-8"))

    if not proposals:
        print(json.dumps({"ok": False, "reason": "no proposals"}, ensure_ascii=False))
        return

    # Load fix plan if in fix mode
    fix_plan = None
    if is_fix_mode:
        fix_path = art_dir / "smoke_fix_plan.json"
        if fix_path.exists():
            fix_plan = json.loads(fix_path.read_text(encoding="utf-8"))

    generated = []
    validated = []
    generated_run_ids = []
    validated_run_ids = []
    failed = []
    failed_run_ids = []
    failed_details = []
    skipped_registry = []

    for cfg in proposals:
        fix_info = None
        fix_type = ""
        arch_name = cfg.get("arch_name")
        if not arch_name:
            continue
        run_id = f"r{round_num:03d}_smoke_{experiment_id(cfg)}"

        # Registered/shared architectures are not allowed to bypass the
        # per-run codegen/review contract.  They may be used as references, but
        # every proposal still needs an isolated generated model file that can be
        # exact-exported, true-channel validated, and later copied to the run's
        # attempt-local model.py.  Keep the cheap registered config/schema guard
        # as an early fail, but never mark a registry arch validated merely
        # because it exists in shared MODEL_REGISTRY.
        if arch_name in _registered_arch_names() and not is_fix_mode:
            reg_ok, reg_msg = validate_registered_arch_config(arch_name, cfg)
            if not reg_ok:
                msg = f"{arch_name}: registered_arch_config ({reg_msg})"
                failed.append(msg)
                failed_run_ids.append(run_id)
                failed_details.append({"run_id": run_id, "arch_name": arch_name, "stage": "registered_arch_config", "message": reg_msg})
                continue
            LOGGER.info("Registered arch %s passed schema guard; generating isolated model for validation", arch_name)

        if is_fix_mode and fix_plan:
            # Fix mode: only fix the specific semantic run selected by the
            # controller/reviewer.  Matching by arch_name is unsafe because one
            # round can contain multiple configs for the same architecture.
            semantic_id = experiment_id(cfg)
            run_id = f"r{round_num:03d}_smoke_{semantic_id}"
            full_run_id = f"r{round_num:03d}_full_{semantic_id}"
            fix_info = None
            for fix in fix_plan.get("fixes", []):
                fid = fix.get("exp_id", "")
                if (
                    fid == run_id or fid.startswith(run_id + "_repair") or fid.startswith(run_id + "_retry")
                    or fid == full_run_id or fid.startswith(full_run_id + "_repair") or fid.startswith(full_run_id + "_retry")
                ):
                    fix_info = fix
                    break
            if not fix_info or not fix_info.get("fixable"):
                continue  # Skip non-fixable/non-target runs

            # Load error log from the exact result source selected by reviewer.
            result_key = "full_results" if fix_plan.get("source_tag") == "full" else "smoke_results"
            diagnosis_results = state.get(result_key, [])
            if not diagnosis_results:
                result_path = art_dir / ("full_results.json" if result_key == "full_results" else "smoke_results.json")
                if result_path.exists():
                    diagnosis_results = json.loads(result_path.read_text(encoding="utf-8"))
            error_log = ""
            for sr in diagnosis_results:
                sid = sr.get("experiment_id") or sr.get("exp_id", "")
                if sid == fix_info.get("exp_id"):
                    error_log = sr.get("log_tail", "") or sr.get("error", "") or (sr.get("metrics", {}) or {}).get("error_message", "")
                    break
            fix_type = str(fix_info.get("fix_type", "code")).lower()
            if fix_type != "code":
                LOGGER.info("Skipping non-code fix for %s (%s)", arch_name, fix_info.get("fix_type"))
                continue

            fix_info["error_log"] = error_log
            fix_info["arch_kwargs"] = cfg.get("arch_kwargs", {})

            gen_ok, gen_msg = fix_model_code(arch_name, fix_info)
        else:
            # Normal mode: generate new code
            gen_ok, gen_msg = generate_model_code(arch_name, cfg)

        if gen_ok:
            generated.append(arch_name)
            generated_run_ids.append(run_id)
            val_ok, val_msg = validate_model_code(arch_name, cfg)

            # Validation feedback loop.  A failed generated file is often very
            # close but missing a helper, shape guard, import, or size guard.
            # Feed the exact validation error and failed generated file back into
            # Codex immediately.  This must run for both first-pass generation
            # and later fix-mode repairs; otherwise initial partial codegen
            # failures block the whole round before the repair machinery starts.
            if not val_ok:
                if is_fix_mode and fix_info and fix_type == "code":
                    base_fix_info = dict(fix_info)
                else:
                    base_fix_info = {
                        "fix_type": "code",
                        "diagnosis": "codegen validation failure",
                        "fix_description": (
                            "Fix the generated model so it passes import, "
                            "instantiation, and forward validation at the configured scale."
                        ),
                        "error_log": "",
                        "arch_kwargs": cfg.get("arch_kwargs", {}),
                        "input_features": cfg.get("input_features", "height"),
                        "n_c": cfg.get("n_c", cfg.get("arch_kwargs", {}).get("n_c", 16)),
                        "depth": cfg.get("depth", cfg.get("arch_kwargs", {}).get("depth", 7)),
                    }

                validation_feedback = []
                for attempt in range(1, 4):
                    feedback_info = dict(base_fix_info)
                    feedback_info["attempt"] = attempt
                    feedback_info.setdefault("input_features", cfg.get("input_features", "height"))
                    feedback_info.setdefault("n_c", cfg.get("n_c", cfg.get("arch_kwargs", {}).get("n_c", 16)))
                    feedback_info.setdefault("depth", cfg.get("depth", cfg.get("arch_kwargs", {}).get("depth", 7)))
                    feedback_info["validation_error"] = val_msg
                    feedback_info["fix_description"] = (
                        str(base_fix_info.get("fix_description", "fix the error"))
                        + "\n\nAdditional validation-feedback task: the previous generated file failed validation. "
                        + "Use the validation error below as the immediate target and output a full self-contained file."
                    )
                    gen_ok2, gen_msg2 = fix_model_code(arch_name, feedback_info)
                    validation_feedback.append({"attempt": attempt, "generated": gen_ok2, "message": gen_msg2, "validation_error": val_msg})
                    if not gen_ok2:
                        gen_msg = gen_msg2
                        break
                    val_ok, val_msg = validate_model_code(arch_name, cfg)
                    if val_ok:
                        gen_ok = True
                        break
                if validation_feedback:
                    gen_msg = f"{gen_msg}; validation_feedback={validation_feedback[-1]}"

            if val_ok:
                validated.append(arch_name)
                validated_run_ids.append(run_id)
            else:
                msg = f"{arch_name}: validation ({val_msg})"
                failed.append(msg)
                failed_run_ids.append(run_id)
                failed_details.append({"run_id": run_id, "arch_name": arch_name, "stage": "validation", "message": val_msg})
        else:
            msg = f"{arch_name}: generation ({gen_msg})"
            failed.append(msg)
            failed_run_ids.append(run_id)
            failed_details.append({"run_id": run_id, "arch_name": arch_name, "stage": "generation", "message": gen_msg})

    # Write manifest
    manifest = {
        "timestamp": now_iso(),
        "mode": "fix" if is_fix_mode else "generate",
        "proposals_count": len(proposals),
        "generated_count": len(generated),
        "validated_count": len(validated),
        "generated_archs": generated,
        "validated_archs": validated,
        "generated_run_ids": generated_run_ids,
        "validated_run_ids": validated_run_ids,
        "skipped_registry_archs": skipped_registry,
        "failed": failed,
        "failed_run_ids": failed_run_ids,
        "failed_details": failed_details,
        "postprocess_actions": POSTPROCESS_ACTIONS,
        "terminal_codegen_failures": list(failed_details),
        "validated": len(failed) == 0,
    }
    (art_dir / "codegen_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Codegen %s: %d generated, %d validated, %d failed",
                "fix" if is_fix_mode else "generate",
                len(generated), len(validated), len(failed))
    print(json.dumps({
        "ok": manifest["validated"],
        "mode": manifest["mode"],
        "generated": len(generated),
        "validated": len(validated),
        "failed": len(failed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    main()






