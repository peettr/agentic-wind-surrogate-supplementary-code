#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

ROOT = Path('<BASELINE_HPC_SOURCE_ROOT>')
WORK = Path('<GRID_HPC_SOURCE_ROOT>/reports/grid18_split_sensitivity_20260507_broad')
RUN_DIR = ROOT / 'campaigns/grid18_200ep/baseline'
CONFIG = RUN_DIR / 'train_config.json'
CKPT = RUN_DIR / 'model_best.pt'
SPLIT_MANIFESTS = {
    'auto_v3': Path('<BASELINE_HPC_SOURCE_ROOT>/shared/data/split_manifest.json'),
    'auto_v5': Path('<GRID_HPC_SOURCE_ROOT>/shared/data/split_manifest.json'),
}
# seed=1 is the current 55/47 split used by Auto11 candidates.
# seed=7 is the historical memory split with 51/51.
# The other seeds with the same counts are included for sensitivity/provenance.
SEEDS = [1, 7, 3, 10, 20, 8, 12, 14]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'shared'))

import torch
from shared.configs.schema import TrainConfig
from shared.train import _resolve_model_cls
from shared.eval_module import EvalModule


def compact_metrics(m: dict) -> dict:
    out = {k: m.get(k) for k in ['r2_median', 'r2_mean', 'r2_global', 'mae_median', 'mae_mean', 'n_cases_evaluated', 'n_fail'] if k in m}
    pc = m.get('per_case_r2') or {}
    out['n_per_case_r2'] = len(pc)
    out['case_ids'] = sorted(pc.keys())
    return out


def main():
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.model_validate(json.loads(CONFIG.read_text()))
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
        'config_path': str(CONFIG),
        'model_path': str(CKPT),
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'config': {
            'arch_name': cfg.arch_name,
            'epochs': cfg.epochs,
            'seed': cfg.seed,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'loss_name': cfg.loss_name,
            'arch_kwargs': dict(cfg.arch_kwargs or {}),
            'original_split_manifest_path': str(cfg.split_manifest_path),
            'data_dir': str(cfg.data_dir),
        },
        'evaluations': [],
    }

    for manifest_label, manifest_path in SPLIT_MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text())
        print('MANIFEST', manifest_label, manifest_path, 'content_hash', manifest.get('content_hash'), flush=True)
        evaluator = EvalModule(str(manifest_path), cfg.data_dir)
        for seed in SEEDS:
            s = manifest['seeds'][str(seed)]
            counts = {'train': len(s['train']), 'val': len(s['val']), 'holdout': len(s['holdout'])}
            if (counts['val'], counts['holdout']) not in [(55,47), (51,51)]:
                continue
            rec = {
                'manifest_label': manifest_label,
                'manifest_path': str(manifest_path),
                'manifest_content_hash': manifest.get('content_hash'),
                'eval_hash': getattr(evaluator, 'eval_hash', ''),
                'split_hash': getattr(evaluator, 'split_hash', ''),
                'seed': seed,
                'counts': counts,
                'split_type': f"{counts['val']}/{counts['holdout']}",
            }
            try:
                val = evaluator.evaluate(model, split='val', seed=seed, batch_size=4, device=device)
                hold = evaluator.evaluate(model, split='holdout', seed=seed, batch_size=4, device=device)
                rec['status'] = 'ok'
                rec['val_metrics'] = compact_metrics(val)
                rec['holdout_metrics'] = compact_metrics(hold)
                print('OK', manifest_label, 'seed', seed, 'split', rec['split_type'],
                      'val_r2med', rec['val_metrics'].get('r2_median'),
                      'hold_r2med', rec['holdout_metrics'].get('r2_median'), flush=True)
            except Exception as e:
                rec['status'] = 'error'
                rec['error'] = repr(e)
                rec['traceback'] = traceback.format_exc()
                print('ERR', manifest_label, 'seed', seed, repr(e), flush=True)
            result['evaluations'].append(rec)
            (WORK / 'grid18_split_sensitivity.partial.json').write_text(json.dumps(result, indent=2))
    result['seconds'] = time.time() - t0
    out = WORK / 'grid18_split_sensitivity.json'
    out.write_text(json.dumps(result, indent=2))
    print('SAVED', out, flush=True)

if __name__ == '__main__':
    main()
