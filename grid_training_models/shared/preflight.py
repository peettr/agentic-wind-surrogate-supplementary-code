"""Preflight: hard gate that prevents invalid code from reaching a GPU.

Six checks, all run on CPU with a ``B=1`` dummy batch (Appendix §2.3):

1. AST syntax parse.
2. Forward pass on ``(1, 1, 640, 640)`` zeros.
3. Dimension check — output must equal ``(1, 1, 640, 640)``.
4. VRAM estimate — ``params × overhead`` must not exceed 80 GB.
5. Differentiability — ``loss.sum().backward()`` on a randn sample.
6. NaN safety — forward output and all parameter gradients finite.

Preflight is stateless and side-effect-free except for module import.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from shared.configs.schema import PreflightCheck, PreflightReport


VRAM_LIMIT_GB = 80.0
INPUT_SHAPE = (1, 1, 640, 640)
EXPECTED_OUTPUT_SHAPE = (1, 1, 640, 640)
# Heuristic bytes per parameter at batch=16 (empirical from v2 7-level UNet).
VRAM_OVERHEAD_PER_PARAM = 48.0
# Activation-memory estimate (fix #17). At 640x640 the activation footprint
# dominates the parameter footprint for UNet-depth>=6 and FNO-wide models.
# Heuristic: batch * channels * H * W * 4 bytes * depth * 2 (fwd + bwd).
DEFAULT_BATCH_FOR_VRAM = 16
DEFAULT_DEPTH_FALLBACK = 5
DEFAULT_CHANNELS_FALLBACK = 16


def estimate_activation_memory_gb(
    batch: int,
    channels: int,
    height: int,
    width: int,
    depth: int,
) -> float:
    """Return an activation-memory estimate in GB (fwd+bwd)."""
    return batch * channels * height * width * 4 * depth * 2 / 1e9


class Preflight:
    """Runs the six mandatory checks on a generated model module."""

    def check(
        self,
        script_path: str | Path,
        class_name: str,
        kwargs: Optional[dict[str, Any]] = None,
        loss_script_path: Optional[str | Path] = None,
        loss_class_name: Optional[str] = None,
        loss_kwargs: Optional[dict[str, Any]] = None,
    ) -> PreflightReport:
        """Preflight an architecture module.

        When ``loss_script_path`` is provided the masked-NaN check uses the
        generated loss class instead of the built-in ``masked_l1``
        (addresses Codex V2 issue 3). Otherwise the masked-NaN check falls
        back to ``masked_l1`` — preserves the previous behavior for arch-only
        proposals.
        """
        t0 = time.time()
        script_path = Path(script_path)
        kwargs = kwargs or {}
        checks: list[PreflightCheck] = []
        vram_est: Optional[float] = None

        # 1. AST parse ----------------------------------------------------
        ok, detail = self._check_ast(script_path)
        checks.append(PreflightCheck(name="ast_parse", passed=ok, detail=detail))
        if not ok:
            return self._fail(script_path, checks, t0)

        # Module import ---------------------------------------------------
        try:
            module = self._import_module(script_path)
            cls: Callable[..., torch.nn.Module] = getattr(module, class_name)
        except Exception as exc:
            checks.append(PreflightCheck(
                name="import", passed=False,
                detail=f"{exc}\n{traceback.format_exc()}",
            ))
            return self._fail(script_path, checks, t0)
        checks.append(PreflightCheck(name="import", passed=True, detail=class_name))

        # 2. Forward pass -------------------------------------------------
        try:
            model = cls(**kwargs)
            model.eval()
            x = torch.zeros(INPUT_SHAPE)
            with torch.no_grad():
                y = model(x)
        except Exception as exc:
            checks.append(PreflightCheck(
                name="forward", passed=False,
                detail=f"{exc}\n{traceback.format_exc()}",
            ))
            return self._fail(script_path, checks, t0)
        checks.append(PreflightCheck(name="forward", passed=True,
                                     detail=f"out shape={tuple(y.shape)}"))

        # 3. Dimension check ---------------------------------------------
        shape_ok = tuple(y.shape) == EXPECTED_OUTPUT_SHAPE
        checks.append(PreflightCheck(
            name="dimensions", passed=shape_ok,
            detail=f"got={tuple(y.shape)} expected={EXPECTED_OUTPUT_SHAPE}",
        ))

        # 4. VRAM estimate (parameters + activations) -------------------
        n_params = sum(p.numel() for p in model.parameters())
        param_bytes_gb = n_params * VRAM_OVERHEAD_PER_PARAM / 1e9
        # Activation memory often dominates at 640x640 (fix #17). Use the
        # model's declared depth and channels if available, else fall back
        # to conservative defaults.
        depth = int(
            kwargs.get("depth")
            or getattr(model, "depth", None)
            or DEFAULT_DEPTH_FALLBACK
        )
        channels = int(
            kwargs.get("n_c")
            or kwargs.get("width")
            or getattr(model, "n_c", None)
            or getattr(model, "width", None)
            or DEFAULT_CHANNELS_FALLBACK
        )
        activation_gb = estimate_activation_memory_gb(
            DEFAULT_BATCH_FOR_VRAM, channels,
            INPUT_SHAPE[2], INPUT_SHAPE[3], depth,
        )
        vram_est = param_bytes_gb + activation_gb
        checks.append(PreflightCheck(
            name="vram", passed=vram_est <= VRAM_LIMIT_GB,
            detail=(
                f"~{vram_est:.2f} GB "
                f"(params={param_bytes_gb:.2f}, activations={activation_gb:.2f}, "
                f"depth={depth}, channels={channels}, limit={VRAM_LIMIT_GB} GB)"
            ),
        ))

        # 5 + 6. Differentiability + NaN safety ------------------------
        try:
            model.train()
            x2 = torch.randn(INPUT_SHAPE)
            y2 = model(x2)
            y2.sum().backward()
            has_nan_fwd = bool(torch.isnan(y2).any() or torch.isinf(y2).any())
            grad_nan = any(
                p.grad is not None
                and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
        except Exception as exc:
            checks.append(PreflightCheck(
                name="differentiability", passed=False,
                detail=f"{exc}\n{traceback.format_exc()}",
            ))
            checks.append(PreflightCheck(
                name="nan_safety", passed=False, detail="backward() raised",
            ))
            return self._fail(script_path, checks, t0, vram_est)

        checks.append(PreflightCheck(name="differentiability", passed=True,
                                     detail="backward OK"))
        checks.append(PreflightCheck(
            name="nan_safety", passed=not (has_nan_fwd or grad_nan),
            detail=f"fwd_nan={has_nan_fwd} grad_nan={grad_nan}",
        ))

        # 7. Masked-loss NaN check (fix #18) -----------------------------
        # Exercise the full loss path with a target that contains NaNs
        # (mimicking the building/invalid pixels that EvalModule masks out).
        # If the loss uses mask*diff instead of torch.where, NaN gradients
        # leak here and this check fails. When a generated loss is
        # supplied, test that one instead of the built-in masked_l1
        # (addresses V2 issue 3).
        loss_fn = self._build_loss_fn(
            loss_script_path, loss_class_name, loss_kwargs or {},
        )
        masked_ok, masked_detail = self._check_masked_loss_nan(model, loss_fn)
        checks.append(PreflightCheck(
            name="masked_loss_nan", passed=masked_ok, detail=masked_detail,
        ))

        return PreflightReport(
            script_path=str(script_path),
            passed=all(c.passed for c in checks),
            checks=checks,
            vram_estimate_gb=vram_est,
            elapsed_sec=time.time() - t0,
        )

    def check_loss(
        self,
        loss_script_path: str | Path,
        loss_class_name: str,
        loss_kwargs: Optional[dict[str, Any]] = None,
    ) -> PreflightReport:
        """Preflight a generated loss module standalone.

        Builds a tiny proxy model and runs the same masked-NaN gradient
        check used for architectures. Catches generated losses that use
        ``mask * diff`` instead of ``torch.where`` (V2 issue 3 / partial 18).
        """
        t0 = time.time()
        checks: list[PreflightCheck] = []
        loss_path = Path(loss_script_path)

        ok, detail = self._check_ast(loss_path)
        checks.append(PreflightCheck(name="ast_parse", passed=ok, detail=detail))
        if not ok:
            return self._fail(loss_path, checks, t0)

        try:
            loss_fn = self._build_loss_fn(
                loss_path, loss_class_name, loss_kwargs or {},
            )
            if loss_fn is None:
                raise ImportError(
                    f"Could not resolve loss class {loss_class_name!r} "
                    f"in {loss_path}"
                )
        except Exception as exc:
            checks.append(PreflightCheck(
                name="import", passed=False,
                detail=f"{exc}\n{traceback.format_exc()}",
            ))
            return self._fail(loss_path, checks, t0)
        checks.append(PreflightCheck(
            name="import", passed=True, detail=loss_class_name,
        ))

        # Minimal trainable proxy: a 1x1 conv so .backward() produces real
        # gradients against the loss without pulling in a full model.
        proxy = torch.nn.Conv2d(1, 1, kernel_size=1)
        masked_ok, masked_detail = self._check_masked_loss_nan(proxy, loss_fn)
        checks.append(PreflightCheck(
            name="masked_loss_nan", passed=masked_ok, detail=masked_detail,
        ))

        return PreflightReport(
            script_path=str(loss_path),
            passed=all(c.passed for c in checks),
            checks=checks,
            vram_estimate_gb=None,
            elapsed_sec=time.time() - t0,
        )

    # ---- helpers --------------------------------------------------------
    def _build_loss_fn(
        self,
        loss_script_path: Optional[str | Path],
        loss_class_name: Optional[str],
        loss_kwargs: dict[str, Any],
    ) -> Optional[torch.nn.Module]:
        """Resolve either a generated loss (via script_path) or masked_l1."""
        if loss_script_path and loss_class_name:
            module = self._import_module(Path(loss_script_path))
            cls = getattr(module, loss_class_name, None)
            if cls is None:
                return None
            return cls(**loss_kwargs)
        try:
            from shared.losses import LIBRARY  # deferred import
        except Exception:  # pragma: no cover
            return None
        return LIBRARY.build("masked_l1")

    @staticmethod
    def _check_masked_loss_nan(
        model: torch.nn.Module,
        loss_fn: Optional[torch.nn.Module] = None,
    ) -> tuple[bool, str]:
        """Exercise the supplied loss with a NaN-containing target.

        Uses the given ``loss_fn``; when ``None`` (legacy call site) falls
        back to the built-in :data:`shared.losses.LIBRARY`'s ``masked_l1``.
        """
        if loss_fn is None:
            try:
                from shared.losses import LIBRARY  # deferred import
            except Exception as exc:  # pragma: no cover
                return False, f"loss library import failed: {exc}"
            loss_fn = LIBRARY.build("masked_l1")
        try:
            loss_label = type(loss_fn).__name__
            model.train()
            x = torch.zeros(INPUT_SHAPE)
            # Negative X = building pixel (enters mask).
            x[:, :, :10, :10] = -1.0
            y = model(x)
            target = torch.randn_like(y)
            # Sprinkle NaNs in the target: they must be safely masked out.
            target[:, :, :5, :5] = float("nan")
            # Also place a NaN in an X-valid region to exercise the mask.
            target[:, :, 100:110, 100:110] = float("nan")
            loss = loss_fn(y, target, x)
            if torch.isnan(loss) or torch.isinf(loss):
                return False, f"loss is nan/inf: {loss.item()!r}"
            loss.backward()
            for p in model.parameters():
                if p.grad is None:
                    continue
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return False, f"NaN in gradients after {loss_label} backward"
            return True, f"{loss_label} NaN-target path OK (loss={loss.item():.4e})"
        except Exception as exc:
            return False, f"{exc}"
        finally:
            # Clean gradients so the caller's subsequent checks start fresh.
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

    @staticmethod
    def _fail(
        script_path: Path,
        checks: list[PreflightCheck],
        t0: float,
        vram_est: Optional[float] = None,
    ) -> PreflightReport:
        return PreflightReport(
            script_path=str(script_path),
            passed=False,
            checks=checks,
            vram_estimate_gb=vram_est,
            elapsed_sec=time.time() - t0,
        )

    @staticmethod
    def _check_ast(path: Path) -> tuple[bool, str]:
        try:
            src = path.read_text()
            ast.parse(src)
            return True, f"{len(src)} chars parsed OK"
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}"
        except FileNotFoundError:
            return False, f"File not found: {path}"

    @staticmethod
    def _import_module(path: Path):
        mod_name = f"preflight_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module


__all__ = ["Preflight", "VRAM_LIMIT_GB"]
