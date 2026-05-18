#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

ROOT = Path('<BASELINE_HPC_SOURCE_ROOT>')
WORK = Path('<GRID_HPC_SOURCE_ROOT>/reports/grid18_seed1_forced51_20260507')
RUN_DIR = ROOT / 'campaigns/grid18_200ep/baseline'
CONFIG = RUN_DIR / 'train_config.json'
CKPT = RUN_DIR / 'model_best.pt'
BASE_MANIFEST = Path('<GRID_HPC_SOURCE_ROOT>/shared/data/split_manifest.json')

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'shared'))

import torch
from shared.configs.schema import TrainConfig
from shared.train import _resolve_model_cls
from shared.eval_module import EvalModule


def write_manifest_variant(name: str, val_cases: list[str], holdout_cases: list[str], train_cases: list[str], base_manifest: dict) -> Path:
    m = json.loads(json.dumps(base_manifest))
    m['seeds']['1']['train'] = list(train_cases)
    m['seeds']['1']['val'] = list(val_cases)
    m['seeds']['1']['holdout'] = list(holdout_cases)
    m['seeds']['1']['n_train'] = len(train_cases)
    m['seeds']['1']['n_val'] = len(val_cases)
    m['seeds']['1']['n_holdout'] = len(holdout_cases)
    m['content_hash'] = f'temporary_{name}_seed1_forced_case_ids_20260507'
    p = WORK / f'split_manifest_{name}.json'
    p.write_text(json.dumps(m, indent=2))
    return p


def compact(m: dict) -> dict:
    pc = m.get('per_case_r2') or {}
    return {
        'r2_median': m.get('r2_median'),
        'r2_mean': m.get('r2_mean'),
        'r2_global': m.get('r2_global'),
        'mae_median': m.get('mae_median'),
        'mae_mean': m.get('mae_mean'),
        'n_cases_evaluated': m.get('n_cases_evaluated'),
        'n_fail': m.get('n_fail'),
        'n_per_case_r2': len(pc),
        'case_ids': sorted(pc.keys()),
    }


def eval_variant(label: str, manifest_path: Path, cfg, model, device):
    ev = EvalModule(str(manifest_path), cfg.data_dir)
    val = ev.evaluate(model, split='val', seed=1, batch_size=4, device=device)
    hold = ev.evaluate(model, split='holdout', seed=1, batch_size=4, device=device)
    return {
        'label': label,
        'manifest_path': str(manifest_path),
        'eval_hash': getattr(ev, 'eval_hash', ''),
        'split_hash': getattr(ev, 'split_hash', ''),
        'seed_arg': 1,
        'val_metrics': compact(val),
        'holdout_metrics': compact(hold),
    }


def main():
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.model_validate(json.loads(CONFIG.read_text()))
    base = json.loads(BASE_MANIFEST.read_text())
    seed1 = base['seeds']['1']
    seed7 = base['seeds']['7']
    seed1_train = list(seed1['train'])
    seed1_test_sorted = sorted(set(seed1['val']) | set(seed1['holdout']))
    current_val_sorted = sorted(seed1['val'])
    current_holdout_sorted = sorted(seed1['holdout'])

    variants = []
    # Current Auto11 split, control
    variants.append(('current_seed1_55_47', BASE_MANIFEST))
    # Exact same 51/51 case IDs that were previously addressed by manifest seed 7, but now under manifest seed 1 and evaluated with seed=1.
    variants.append(('seed1_using_exact_seed7_51_51_case_ids', write_manifest_variant(
        'seed1_exact_seed7_51_51_case_ids', seed7['val'], seed7['holdout'], seed1_train, base)))
    # Fair same-train/test 51/51: only repartition seed=1 test union by sorted IDs.
    variants.append(('seed1_test_union_sorted_51_51', write_manifest_variant(
        'seed1_test_union_sorted_51_51', seed1_test_sorted[:51], seed1_test_sorted[51:], seed1_train, base)))
    # Fair same-train/test 51/51 with minimal departure from current 55/47: remove the last four sorted current-val cases to holdout.
    variants.append(('seed1_current_val_trimmed_to_51_plus_51', write_manifest_variant(
        'seed1_current_val_trimmed_51_51', current_val_sorted[:51], sorted(current_val_sorted[51:] + current_holdout_sorted), seed1_train, base)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = _resolve_model_cls(cfg)
    model = model_cls(**cfg.arch_kwargs).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    result = {
        'baseline_name': 'grid18_200ep_baseline',
        'run_dir': str(RUN_DIR),
        'model_path': str(CKPT),
        'config_path': str(CONFIG),
        'base_manifest': str(BASE_MANIFEST),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'variant_notes': {
            'current_seed1_55_47': 'Original Auto11 seed=1 split, 55 val and 47 holdout.',
            'seed1_using_exact_seed7_51_51_case_ids': 'Manifest seed=1 is overwritten with the exact case IDs from manifest seed=7 val/holdout, then evaluator is called with seed=1. This answers same IDs but seed_arg=1.',
            'seed1_test_union_sorted_51_51': 'Fair seed=1 train/test, using only current seed=1 test union, repartitioned sorted 51/51.',
            'seed1_current_val_trimmed_to_51_plus_51': 'Fair seed=1 train/test, current 55 val trimmed to 51 and moved four val cases into holdout.',
        },
        'evaluations': [],
    }
    train1 = set(seed1_train)
    for label, mp in variants:
        try:
            rec = eval_variant(label, Path(mp), cfg, model, device)
            rec['val_cases_in_seed1_train'] = len(set(rec['val_metrics']['case_ids']) & train1)
            rec['holdout_cases_in_seed1_train'] = len(set(rec['holdout_metrics']['case_ids']) & train1)
            rec['status'] = 'ok'
            print('OK', label,
                  'val_n', rec['val_metrics']['n_cases_evaluated'], 'hold_n', rec['holdout_metrics']['n_cases_evaluated'],
                  'val_r2med', rec['val_metrics']['r2_median'], 'hold_r2med', rec['holdout_metrics']['r2_median'],
                  'val_train_overlap', rec['val_cases_in_seed1_train'], 'hold_train_overlap', rec['holdout_cases_in_seed1_train'], flush=True)
        except Exception as e:
            rec = {'label': label, 'manifest_path': str(mp), 'status': 'error', 'error': repr(e), 'traceback': traceback.format_exc()}
            print('ERR', label, repr(e), flush=True)
        result['evaluations'].append(rec)
        (WORK / 'grid18_seed1_forced51.partial.json').write_text(json.dumps(result, indent=2))
    result['seconds'] = time.time() - t0
    out = WORK / 'grid18_seed1_forced51.json'
    out.write_text(json.dumps(result, indent=2))
    print('SAVED', out, flush=True)

if __name__ == '__main__':
    main()
