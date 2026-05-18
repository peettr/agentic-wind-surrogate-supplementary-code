"""V4-only structured knowledge artifacts for planner/reviewer.

All artifacts generated here are derived only from the current V4 campaign state,
results, reviews, and manifests.  V3 performance data must never enter these
files.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from workflow_common import round_artifact_dir, now_iso
from attempt_manifest import base_run_id, load_manifest

FAILURE_CLASSES = {
    "PASS",
    "RESOURCE_OOM",
    "RESOURCE_HELD",
    "RESOURCE_EVICTED",
    "CODEGEN_BUG",
    "SCHEMA_MISMATCH",
    "SHAPE_ERROR",
    "OPTIMIZATION_UNSTABLE",
    "OVERFIT",
    "UNDERFIT",
    "BENCHMARK_TIE",
    "LOW_VALUE_VARIANT",
    "AUTO_FAIL",
    "UNKNOWN",
}


def knowledge_dir(campaign_dir: Path) -> Path:
    p = campaign_dir / "knowledge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _round_from_exp_id(exp_id: str) -> int | None:
    m = re.match(r"r(\d{3})_", exp_id or "")
    return int(m.group(1)) if m else None


def _tag_from_exp_id(exp_id: str) -> str:
    if "_full_" in (exp_id or ""):
        return "full"
    if "_smoke_" in (exp_id or ""):
        return "smoke"
    return "unknown"


def _retry_count(exp_id: str) -> int:
    return 1 if re.search(r"_retry\d+$", exp_id or "") else 0


def _repair_count(exp_id: str) -> int:
    return 1 if re.search(r"_repair\d+$", exp_id or "") else 0


def _failure_class(result: dict[str, Any]) -> str:
    status = str(result.get("status", "")).lower()
    if status == "completed" and result.get("metrics"):
        return "PASS"
    text = "\n".join(
        str(result.get(k, "")) for k in ("classification", "error", "log_tail", "diagnosis")
    ).lower()
    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict):
        text += "\n" + str(metrics.get("error_message", "")).lower()
    if "auto_fail" in text or status.startswith("auto"):
        return "AUTO_FAIL"
    if "cuda out of memory" in text or "outofmemory" in text or "oom" in text or "cgroup memory" in text:
        return "RESOURCE_OOM"
    if status == "held" or "holdreason" in text or "job was held" in text:
        return "RESOURCE_HELD"
    if status == "evicted" or "evicted" in text or "preempted" in text:
        return "RESOURCE_EVICTED"
    if "schema" in text or "unexpected keyword" in text or "validation error" in text:
        return "SCHEMA_MISMATCH"
    if "shape" in text or "size mismatch" in text or "mat1 and mat2" in text:
        return "SHAPE_ERROR"
    if "nan" in text or status == "loss_nan":
        return "OPTIMIZATION_UNSTABLE"
    if "traceback" in text or "nameerror" in text or "typeerror" in text or "importerror" in text:
        return "CODEGEN_BUG"
    return "UNKNOWN"


def _metric(result: dict[str, Any], key: str) -> float | None:
    if result.get(key) is not None:
        try:
            return float(result[key])
        except Exception:
            return None
    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict):
        val = metrics.get("val_metrics", {}) if isinstance(metrics.get("val_metrics"), dict) else {}
        candidates = {
            "val_r2_median": [val.get("r2_median"), metrics.get("r2_median")],
            "val_mae": [val.get("mae"), metrics.get("mae")],
        }.get(key, [])
        for c in candidates:
            if c is not None:
                try:
                    return float(c)
                except Exception:
                    pass
    return None


def _result_record(result: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    exp_id = result.get("exp_id") or result.get("experiment_id") or ""
    cfg = result.get("config") or {}
    arch_kwargs = cfg.get("arch_kwargs") or {}
    base_id = base_run_id(exp_id) if exp_id else ""
    entry = (manifest or {}).get("runs", {}).get(base_id, {})
    metrics = result.get("metrics") or {}
    params = metrics.get("model_params") if isinstance(metrics, dict) else None
    params_million = metrics.get("params_million") if isinstance(metrics, dict) else None
    if params_million is None and params is not None:
        try:
            params_million = float(params) / 1_000_000.0
        except Exception:
            params_million = None
    return {
        "round": _round_from_exp_id(exp_id),
        "tag": _tag_from_exp_id(exp_id),
        "exp_id": exp_id,
        "base_id": base_id,
        "arch_name": result.get("arch_name") or cfg.get("arch_name"),
        "family": result.get("arch_name") or cfg.get("arch_name"),
        "n_c": cfg.get("n_c", arch_kwargs.get("n_c")),
        "depth": cfg.get("depth", arch_kwargs.get("depth")),
        "lr": cfg.get("lr"),
        "loss_name": cfg.get("loss_name"),
        "batch_size": cfg.get("batch_size"),
        "status": result.get("status"),
        "val_r2_median": _metric(result, "val_r2_median"),
        "val_mae": _metric(result, "val_mae"),
        "failure_class": _failure_class(result),
        "retry_count": int(entry.get("retry_count", 0) or 0) if entry else _retry_count(exp_id),
        "repair_count": int(entry.get("repair_count", 0) or 0) if entry else _repair_count(exp_id),
        "cluster_id": str(result.get("cluster_id") or "") or None,
        "resource_class": "high_vram" if _failure_class(result).startswith("RESOURCE") else "normal",
        "model_params": params,
        "params_million": params_million,
    }


def collect_all_results(campaign_dir: Path) -> list[dict[str, Any]]:
    artifacts = campaign_dir / "artifacts"
    if not artifacts.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for rdir in sorted(artifacts.glob("r[0-9][0-9][0-9]")):
        for name in ("smoke_results.json", "full_results.json"):
            data = _read_json(rdir / name)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        exp_id = row.get("exp_id") or row.get("experiment_id")
                        if exp_id:
                            by_id[exp_id] = row
    return list(by_id.values())


def build_all_results_summary(campaign_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(campaign_dir)
    rows = [_result_record(r, manifest) for r in collect_all_results(campaign_dir)]
    rows.sort(key=lambda r: ((r.get("round") or -1), r.get("tag") or "", r.get("exp_id") or ""))
    kdir = knowledge_dir(campaign_dir)
    (kdir / "all_results_summary.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return rows


def build_family_summaries(campaign_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("family"):
            families[str(r["family"])].append(r)
    summary: dict[str, Any] = {}
    for fam, items in sorted(families.items()):
        full = [r for r in items if r.get("tag") == "full"]
        completed = [r for r in full if r.get("status") == "completed" and r.get("val_r2_median") is not None]
        vals = sorted(float(r["val_r2_median"]) for r in completed)
        fail_counts = Counter(r.get("failure_class") for r in items if r.get("failure_class") != "PASS")
        summary[fam] = {
            "attempts": len(items),
            "full_completed": len(completed),
            "best_val_r2_median": max(vals) if vals else None,
            "median_val_r2_median": vals[len(vals)//2] if vals else None,
            "failure_counts": dict(fail_counts),
            "oom_count": int(fail_counts.get("RESOURCE_OOM", 0)),
            "retry_attempts_observed": sum(1 for r in items if "_retry" in (r.get("exp_id") or "")),
        }
    kdir = knowledge_dir(campaign_dir)
    (kdir / "arch_family_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    failure_summary = Counter(r.get("failure_class") for r in rows if r.get("failure_class") != "PASS")
    by_family_failure: dict[str, dict[str, int]] = {}
    for fam, items in families.items():
        by_family_failure[fam] = dict(Counter(r.get("failure_class") for r in items if r.get("failure_class") != "PASS"))
    (kdir / "failure_taxonomy_summary.json").write_text(
        json.dumps({"overall": dict(failure_summary), "by_family": by_family_failure}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _load_prior_hypothesis_registry(campaign_dir: Path) -> dict[str, Any]:
    data = _read_json(knowledge_dir(campaign_dir) / "hypothesis_registry.json")
    if isinstance(data, dict):
        data.setdefault("open", [])
        data.setdefault("closed", [])
        data.setdefault("resolution_log", [])
        return data
    return {"open": [], "closed": [], "resolution_log": []}


def _extract_knowledge_update(review: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(review, dict):
        return {}
    if isinstance(review.get("knowledge_update"), dict):
        return review["knowledge_update"]
    for key in ("claude", "codex"):
        src = review.get(key)
        if isinstance(src, dict) and isinstance(src.get("knowledge_update"), dict):
            return src["knowledge_update"]
    return {}


def build_experiment_knowledge(campaign_dir: Path, state: dict[str, Any], round_num: int, review: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = build_all_results_summary(campaign_dir)
    family_summary = build_family_summaries(campaign_dir, rows)
    art_dir = round_artifact_dir(campaign_dir, round_num)
    round_full = [r for r in rows if r.get("round") == round_num and r.get("tag") == "full"]
    completed = [r for r in round_full if r.get("status") == "completed" and r.get("val_r2_median") is not None]
    best = max(completed, key=lambda r: r.get("val_r2_median") or -999) if completed else None
    update = _extract_knowledge_update(review)

    failure_taxonomy = update.get("failure_taxonomy") if isinstance(update.get("failure_taxonomy"), list) else []
    if not failure_taxonomy:
        failure_taxonomy = [
            {
                "id": f"F{round_num:03d}-{i+1:02d}",
                "arch_name": r.get("arch_name"),
                "failure_class": r.get("failure_class"),
                "evidence_run_ids": [r.get("exp_id")],
                "planner_implication": "Treat as V4-only failure evidence for future resource/model planning.",
            }
            for i, r in enumerate(round_full)
            if r.get("failure_class") not in {None, "PASS"}
        ]

    positive_patterns = update.get("positive_patterns") if isinstance(update.get("positive_patterns"), list) else []
    if not positive_patterns and best:
        positive_patterns = [{
            "id": f"P{round_num:03d}-01",
            "pattern": f"Best completed full result this round: {best.get('arch_name')}",
            "evidence_run_ids": [best.get("exp_id")],
            "planner_implication": "Use as V4-only current strong pattern or control candidate.",
        }]

    knowledge = {
        "round": round_num,
        "timestamp": now_iso(),
        "best_current": best,
        "benchmark_status": update.get("benchmark_survival_explanation", {}) if isinstance(update.get("benchmark_survival_explanation"), dict) else {
            "summary": update.get("benchmark_survival_explanation") or "Not assessed in structured form.",
            "cited_failed_attempts": [],
        },
        "failure_taxonomy": failure_taxonomy,
        "positive_patterns": positive_patterns,
        "negative_patterns": update.get("negative_patterns", []),
        "new_hypotheses": update.get("recommended_hypotheses", []),
        "hypothesis_resolution_log": update.get("hypothesis_resolution_log", []),
        "cooldowns": update.get("cooldowns", []),
        "family_summary_excerpt": family_summary,
    }
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "experiment_knowledge.json").write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")

    registry = _load_prior_hypothesis_registry(campaign_dir)
    existing_open_ids = {h.get("id") for h in registry.get("open", []) if isinstance(h, dict)}
    for h in knowledge.get("new_hypotheses", []) or []:
        if isinstance(h, dict):
            hid = h.get("id") or f"H{round_num:03d}-{len(registry['open'])+1:02d}"
            if hid not in existing_open_ids:
                item = {**h, "id": hid, "opened_round": round_num, "deadline_round": round_num + 3}
                registry["open"].append(item)
                existing_open_ids.add(hid)
    for log in knowledge.get("hypothesis_resolution_log", []) or []:
        if isinstance(log, dict):
            registry["resolution_log"].append({"round": round_num, **log})
            hid = log.get("hypothesis_id") or log.get("id")
            outcome = log.get("outcome") or log.get("action")
            if hid and outcome in {"supported", "refuted", "inconclusive", "abandoned", "closed"}:
                moved = None
                keep = []
                for h in registry.get("open", []):
                    if h.get("id") == hid:
                        moved = h
                    else:
                        keep.append(h)
                registry["open"] = keep
                registry["closed"].append({**(moved or {"id": hid}), "outcome": outcome, "closed_round": round_num, "evidence": log.get("evidence", [])})
    kdir = knowledge_dir(campaign_dir)
    (kdir / "hypothesis_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    build_recent_knowledge_bundle(campaign_dir)
    build_knowledge_summary(campaign_dir, rows, registry)
    return knowledge


def build_recent_knowledge_bundle(campaign_dir: Path, n: int = 5) -> dict[str, Any]:
    artifacts = campaign_dir / "artifacts"
    bundles = []
    if artifacts.exists():
        for rdir in sorted(artifacts.glob("r[0-9][0-9][0-9]"))[-n:]:
            data = _read_json(rdir / "experiment_knowledge.json")
            if data is not None:
                bundles.append(data)
    out = {"rounds": bundles}
    (knowledge_dir(campaign_dir) / "recent_knowledge_bundle.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def build_knowledge_summary(campaign_dir: Path, rows: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    best = [r for r in rows if r.get("tag") == "full" and r.get("status") == "completed" and r.get("val_r2_median") is not None]
    best_sorted = sorted(best, key=lambda r: r.get("val_r2_median") or -999, reverse=True)[:10]
    summary = {
        "timestamp": now_iso(),
        "total_results": len(rows),
        "top_full_results": best_sorted,
        "failure_counts": dict(Counter(r.get("failure_class") for r in rows if r.get("failure_class") != "PASS")),
        "open_hypotheses_count": len(registry.get("open", [])),
        "closed_hypotheses_count": len(registry.get("closed", [])),
    }
    (knowledge_dir(campaign_dir) / "knowledge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def rebuild_knowledge(campaign_dir: Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or {}
    rows = build_all_results_summary(campaign_dir)
    build_family_summaries(campaign_dir, rows)
    registry = _load_prior_hypothesis_registry(campaign_dir)
    build_recent_knowledge_bundle(campaign_dir)
    return build_knowledge_summary(campaign_dir, rows, registry)
