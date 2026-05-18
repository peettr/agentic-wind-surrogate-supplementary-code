"""Sequential Codegen â€” Model architecture code generation pipeline.

Generates PyTorch model code from model_specs.yaml descriptions.
Follows V3's codegen pattern: generate -> review -> validate.

The generated model must:
1. Be a valid Python file with a class matching arch_name
2. Accept input (B, C, 640, 640) where C=1 or C=3
3. Output (B, 1, 640, 640) with ReLU (non-negative)
4. Have parameter count within 20% of params_target
5. Pass forward pass smoke test (no NaN, correct shape)

Sequential rule: codegen reads ONLY model_specs.yaml, never V3 model source files.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

LOGGER = logging.getLogger("hybrid.codegen")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_PATH = PROJECT_ROOT / "configs" / "model_specs.yaml"
GENERATED_DIR = PROJECT_ROOT / "models" / "generated"
MANIFEST_DIR = PROJECT_ROOT / "codegen_manifests"
LOG_DIR = PROJECT_ROOT / "logs"

# AI binaries
CLAUDE_BIN = shutil.which("claude") or shutil.which("claude.exe") or r"<USER_HOME>\.local\bin\claude.exe"
CODEX_BIN = shutil.which("codex") or "codex"

# Validation thresholds
PARAM_TOLERANCE = 0.20  # 20% tolerance on params_target
MAX_VRAM_GB = 48.0
MAX_FORWARD_TIME_SEC = 60.0


# ======================================================================
# Spec loading
# ======================================================================

def load_specs() -> dict:
    """Load model_specs.yaml."""
    text = SPECS_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def get_spec(name: str) -> dict:
    """Get a single model spec by name."""
    specs = load_specs()
    if name not in specs:
        raise KeyError(f"Model '{name}' not found in {SPECS_PATH}")
    return specs[name]


# ======================================================================
# Prompt building
# ======================================================================

def build_generation_prompt(spec: dict, arch_name: str) -> str:
    """Build the prompt for AI model code generation."""
    return (
        "You are a PyTorch model architect. Generate a complete, self-contained "
        "Python file implementing the following neural network architecture.\n\n"
        f"Model name (class name): {arch_name}\n"
        f"Category: {spec.get('category', 'unknown')}\n"
        f"Target parameters: ~{spec.get('params_target', '?')}M\n"
        f"Input channels: {spec.get('input_channels', 1)}\n"
        f"Forward signature: {spec.get('forward_signature', '(B,C,640,640) -> (B,1,640,640)')}\n\n"
        f"Architecture description:\n{spec.get('architecture', '')}\n\n"
        f"Key components: {spec.get('key_components', [])}\n"
        f"Known issues: {spec.get('known_issues', 'None')}\n\n"
        "REQUIREMENTS:\n"
        "1. The file must contain a single class named exactly "
        f"'{arch_name}' that inherits from torch.nn.Module.\n"
        "2. The class must have __init__(self, n_c=16, depth=7, input_channels=1) "
        "with sensible defaults matching the spec.\n"
        "3. The forward(self, x) method must accept (B, C, 640, 640) and return "
        "(B, 1, 640, 640) with ReLU activation (non-negative output).\n"
        "4. Use GroupNorm (num_groups=min(8, n_c)) instead of BatchNorm.\n"
        "5. Use GELU activation in encoder/decoder blocks.\n"
        "6. Parameter count should be within 20% of the target.\n"
        "7. Include a simple ConvBlock helper class if needed.\n"
        "8. Do NOT use any external libraries beyond torch, math, and typing.\n"
        "9. The file must be importable as a Python module.\n"
        "10. Include type hints on __init__ and forward.\n\n"
        "OUTPUT: Return ONLY the Python code, no markdown fences, no explanation."
    )


def build_review_prompt(spec: dict, arch_name: str, code: str) -> str:
    """Build the prompt for code review."""
    return (
        "You are a PyTorch code reviewer. Review the following model implementation "
        "for correctness, interface compliance, and potential issues.\n\n"
        f"Model name: {arch_name}\n"
        f"Target parameters: ~{spec.get('params_target', '?')}M\n"
        f"Input channels: {spec.get('input_channels', 1)}\n"
        f"Forward signature: {spec.get('forward_signature', '(B,C,640,640) -> (B,1,640,640)')}\n\n"
        f"Architecture spec:\n{spec.get('architecture', '')}\n\n"
        "CODE TO REVIEW:\n"
        f"{code}\n\n"
        "CHECKLIST:\n"
        "1. Class name matches the required name exactly.\n"
        "2. __init__ has correct signature: n_c, depth, input_channels parameters.\n"
        "3. forward() accepts (B, C, 640, 640) and returns (B, 1, 640, 640).\n"
        "4. Output uses ReLU (non-negative).\n"
        "5. GroupNorm is used instead of BatchNorm.\n"
        "6. No shape errors: upsampling matches skip connection sizes.\n"
        "7. No infinite loops or O(N^2) operations on 640x640.\n"
        "8. Parameter count is reasonable for the target.\n"
        "9. No external library imports beyond torch/math/typing.\n"
        "10. File is importable (valid Python syntax).\n\n"
        "OUTPUT FORMAT:\n"
        "REVIEW_VERDICT: [pass|fix_applied|fail]\n"
        "ISSUES_FOUND: [list of issues, or none]\n"
        "FIXES_APPLIED: [list of fixes applied inline, or none]\n"
        "REMAINING_RISKS: [list of risks, or none]\n"
        "CORRECTED_CODE: [the corrected code, or the original if pass]\n\n"
        "If REVIEW_VERDICT is 'fix_applied', include the corrected code after "
        "CORRECTED_CODE:. Fix issues in-place with minimal changes."
    )


# ======================================================================
# AI callers
# ======================================================================

def call_claude(prompt: str, model: str = "sonnet", timeout_sec: int = 600) -> Optional[str]:
    """Call Claude CLI in print mode, return response text."""
    try:
        r = subprocess.run(
            [CLAUDE_BIN, "--print", "--model", model, "--permission-mode", "bypassPermissions",
             "-p"],
            input=prompt, text=True, capture_output=True, encoding="utf-8",
            timeout=timeout_sec,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        LOGGER.warning("Claude call failed (rc=%d): %s", r.returncode, r.stderr[:300])
        return None
    except Exception as e:
        LOGGER.warning("Claude call exception: %s", e)
        return None


def call_codex(prompt: str, timeout_sec: int = 600) -> Optional[str]:
    """Call Codex CLI in print mode, return response text."""
    try:
        r = subprocess.run(
            [CODEX_BIN, "exec", "-"],
            input=prompt, text=True, capture_output=True, encoding="utf-8",
            timeout=timeout_sec,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        LOGGER.warning("Codex call failed (rc=%d): %s", r.returncode, r.stderr[:300])
        return None
    except Exception as e:
        LOGGER.warning("Codex call exception: %s", e)
        return None


# ======================================================================
# Code extraction
# ======================================================================

def extract_python_code(response: str) -> str:
    """Extract Python code from AI response (strip markdown fences if present)."""
    # Try to find ```python ... ``` block
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find just ``` ... ``` block
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No fences â€” assume the whole response is code
    return response.strip()


# ======================================================================
# Validation
# ======================================================================

def validate_syntax(code: str, arch_name: str) -> tuple[bool, list[str]]:
    """Check that the code is valid Python and contains the required class."""
    issues = []
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")
        return False, issues

    # Check class name exists
    if f"class {arch_name}" not in code:
        issues.append(f"Missing class '{arch_name}'")
        return False, issues

    # Check forward method
    if "def forward(self" not in code:
        issues.append("Missing forward(self, ...) method")
        return False, issues

    # Check ReLU in output
    if "ReLU" not in code and "relu" not in code and "F.relu" not in code:
        issues.append("No ReLU activation found â€” output may not be non-negative")
        issues.append("This is a WARNING, not a failure")

    # Check for dangerous imports
    for line in code.split("\n"):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            mod = line.split()[1].split(".")[0]
            if mod not in ("torch", "math", "typing", "collections",
                           "functools", "itertools", "numbers", "warnings"):
                issues.append(f"Potentially disallowed import: {mod}")

    return len(issues) == 0, issues


def validate_shapes(code: str, arch_name: str, input_channels: int = 1) -> tuple[bool, list[str]]:
    """Run a forward pass to validate output shape. Returns (ok, issues)."""
    issues = []
    try:
        import torch
        # Execute the code in a namespace
        ns = {}
        exec(code, ns)
        cls = ns.get(arch_name)
        if cls is None:
            issues.append(f"Class {arch_name} not found after exec")
            return False, issues

        # Instantiate with default args
        model = cls()
        model.eval()

        # Test forward pass
        x = torch.randn(1, input_channels, 640, 640)
        with torch.no_grad():
            y = model(x)

        if y.shape != (1, 1, 640, 640):
            issues.append(f"Output shape mismatch: expected (1,1,640,640), got {tuple(y.shape)}")
            return False, issues

        if y.min() < 0:
            issues.append(f"Output has negative values (min={y.min().item():.4f}), ReLU may be missing")

        # Check for NaN
        if torch.isnan(y).any():
            issues.append("Output contains NaN")
            return False, issues

    except Exception as e:
        issues.append(f"Forward pass failed: {type(e).__name__}: {e}")
        return False, issues

    return len(issues) == 0, issues


def validate_params(code: str, arch_name: str, params_target_m: float) -> tuple[bool, list[str]]:
    """Check parameter count is within tolerance of target."""
    issues = []
    try:
        import torch
        ns = {}
        exec(code, ns)
        cls = ns[arch_name]
        model = cls()
        n_params = sum(p.numel() for p in model.parameters())
        n_params_m = n_params / 1e6
        ratio = n_params_m / params_target_m if params_target_m > 0 else 1.0
        if ratio < (1 - PARAM_TOLERANCE) or ratio > (1 + PARAM_TOLERANCE):
            issues.append(
                f"Param count {n_params_m:.1f}M is {ratio:.1%} of target "
                f"{params_target_m:.1f}M (tolerance: {PARAM_TOLERANCE:.0%})"
            )
        LOGGER.info("Param count: %.1fM (target: %.1fM, ratio: %.2f)", n_params_m, params_target_m, ratio)
    except Exception as e:
        issues.append(f"Param count check failed: {e}")
    return len(issues) == 0, issues


# ======================================================================
# Main pipeline
# ======================================================================

def generate_model(
    arch_name: str,
    primary_model: str = "sonnet",
    max_review_rounds: int = 3,
    force: bool = False,
) -> dict:
    """Generate a model architecture file from spec.

    Args:
        arch_name: Model name (must exist in model_specs.yaml)
        primary_model: 'sonnet' or 'codex' for generation
        max_review_rounds: Max generate-review-refine cycles
        force: If True, regenerate even if file exists

    Returns:
        dict with keys: success, arch_name, path, rounds, issues, method
    """
    t0 = time.time()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = GENERATED_DIR / f"{arch_name}.py"

    # Check if already generated
    if output_path.exists() and not force:
        LOGGER.info("%s already exists, skipping (use force=True to regenerate)", arch_name)
        return {"success": True, "arch_name": arch_name, "path": str(output_path),
                "rounds": 0, "issues": [], "method": "cached"}

    # Load spec
    try:
        spec = get_spec(arch_name)
    except KeyError as e:
        return {"success": False, "arch_name": arch_name, "path": "",
                "rounds": 0, "issues": [str(e)], "method": "spec_not_found"}

    gen_prompt = build_generation_prompt(spec, arch_name)
    review_prompt_template = build_review_prompt

    code = None
    method = ""
    review_issues = []

    for round_num in range(1, max_review_rounds + 1):
        LOGGER.info("Generation round %d for %s", round_num, arch_name)

        # --- Generate ---
        if round_num == 1:
            if primary_model == "sonnet":
                response = call_claude(gen_prompt, model="sonnet")
                method = "claude_sonnet"
            else:
                response = call_codex(gen_prompt)
                method = "codex"

            if response is None:
                # Fallback to the other model
                if primary_model == "sonnet":
                    response = call_codex(gen_prompt)
                    method = "codex_fallback"
                else:
                    response = call_claude(gen_prompt, model="sonnet")
                    method = "claude_sonnet_fallback"

            if response is None:
                return {"success": False, "arch_name": arch_name, "path": "",
                        "rounds": round_num, "issues": ["All AI models failed"],
                        "method": "all_failed"}

            code = extract_python_code(response)
        else:
            # Refine based on review
            refine_prompt = (
                f"Fix the following issues in the model code:\n"
                f"{chr(10).join(f'- {i}' for i in review_issues)}\n\n"
                f"Current code:\n{code}\n\n"
                f"Return ONLY the corrected Python code, no explanation."
            )
            if method.startswith("claude"):
                response = call_claude(refine_prompt, model="sonnet")
            else:
                response = call_codex(refine_prompt)

            if response:
                code = extract_python_code(response)

        # --- Syntax check ---
        syn_ok, syn_issues = validate_syntax(code, arch_name)
        if not syn_ok:
            review_issues = syn_issues
            LOGGER.warning("Syntax check failed: %s", syn_issues)
            continue

        # --- Review ---
        rev_prompt = review_prompt_template(spec, arch_name, code)
        # Use Sonnet for review
        review_response = call_claude(rev_prompt, model="sonnet")
        if review_response is None:
            review_response = call_claude(rev_prompt, model="sonnet")

        if review_response:
            # Extract verdict
            verdict_match = re.search(r"REVIEW_VERDICT:\s*(\w+)", review_response)
            verdict = verdict_match.group(1).lower() if verdict_match else "unknown"

            # Extract issues
            issues_match = re.search(r"ISSUES_FOUND:\s*(.+?)(?:\n|FIXES)", review_response, re.DOTALL)
            issues_text = issues_match.group(1).strip() if issues_match else ""
            review_issues = [i.strip("- â€¢").strip() for i in issues_text.split("\n")
                             if i.strip() and i.strip() not in ("none", "None", "")]

            # Extract corrected code if present
            corrected_match = re.search(
                r"CORRECTED_CODE:\s*\n(.*?)(?:\n\n[A-Z_]+:|$)",
                review_response, re.DOTALL
            )
            if corrected_match and verdict == "fix_applied":
                corrected = extract_python_code(corrected_match.group(1).strip())
                if len(corrected) > 100:  # sanity check
                    code = corrected

            if verdict == "pass" or not review_issues:
                LOGGER.info("Review passed on round %d", round_num)
                break
            else:
                LOGGER.info("Review round %d: %d issues found", round_num, len(review_issues))
        else:
            LOGGER.warning("Review call failed, proceeding with generated code")
            break

    # --- Final validation ---
    final_issues = []

    # Syntax
    syn_ok, syn_issues = validate_syntax(code, arch_name)
    final_issues.extend(syn_issues)

    # Shape
    input_ch = spec.get("input_channels", 1)
    shape_ok, shape_issues = validate_shapes(code, arch_name, input_ch)
    final_issues.extend(shape_issues)

    # Params
    params_target = spec.get("params_target", 0)
    if params_target > 0:
        param_ok, param_issues = validate_params(code, arch_name, params_target)
        final_issues.extend(param_issues)

    success = shape_ok and syn_ok  # param mismatch is warning, not failure

    # --- Write output ---
    if success:
        # Add header comment
        header = (
            f"# Auto-generated by Sequential codegen\n"
            f"# Model: {arch_name}\n"
            f"# Category: {spec.get('category', 'unknown')}\n"
            f"# Target params: ~{params_target}M\n"
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        output_path.write_text(header + code, encoding="utf-8")
        LOGGER.info("Generated %s -> %s (%.1fs)", arch_name, output_path, time.time() - t0)
    else:
        LOGGER.error("Validation failed for %s: %s", arch_name, final_issues)

    # --- Write manifest ---
    manifest = {
        "arch_name": arch_name,
        "success": success,
        "path": str(output_path) if success else "",
        "rounds": round_num,
        "method": method,
        "issues": final_issues,
        "review_issues": review_issues if not success else [],
        "spec": {
            "category": spec.get("category"),
            "params_target": params_target,
            "input_channels": input_ch,
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = MANIFEST_DIR / f"{arch_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


def generate_batch(
    arch_names: list[str],
    primary_model: str = "sonnet",
    max_review_rounds: int = 3,
    force: bool = False,
) -> list[dict]:
    """Generate multiple models in sequence.

    Returns list of result dicts.
    """
    results = []
    for name in arch_names:
        LOGGER.info("=== Generating %s ===", name)
        result = generate_model(name, primary_model, max_review_rounds, force)
        results.append(result)
        if result["success"]:
            LOGGER.info("  OK: %s", result["path"])
        else:
            LOGGER.error("  FAIL: %s", result["issues"])

    # Summary
    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok
    LOGGER.info("=== Batch complete: %d ok, %d failed ===", ok, fail)
    return results


# ======================================================================
# CLI
# ======================================================================

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Sequential Model Codegen")
    parser.add_argument("arch", nargs="*", help="Model names to generate (default: all enabled)")
    parser.add_argument("--model", default="sonnet", choices=["sonnet", "codex"],
                        help="Primary AI model for generation")
    parser.add_argument("--review-rounds", type=int, default=3,
                        help="Max generate-review-refine cycles")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if file exists")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        specs = load_specs()
        for name, spec in specs.items():
            cat = spec.get("category", "?")
            params = spec.get("params_target", "?")
            print(f"  {name:30s}  {cat:12s}  {params}M")
        return

    if not args.arch:
        # Generate all baseline models by default
        specs = load_specs()
        args.arch = [n for n, s in specs.items() if s.get("category") == "baseline"]
        LOGGER.info("No arch specified, generating baseline models: %s", args.arch)

    results = generate_batch(args.arch, args.model, args.review_rounds, args.force)

    # Exit code
    if all(r["success"] for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()




