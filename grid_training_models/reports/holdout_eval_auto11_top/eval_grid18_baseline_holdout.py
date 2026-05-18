#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

ROOT = Path('<BASELINE_HPC_SOURCE_ROOT>')
WORK = Path('<GRID_HPC_SOURCE_ROOT>/reports/holdout_eval_auto11_top')
RUN_DIR = ROOT / 'campaigns/grid18_200ep/baseline'
CONFIG = RUN_DIR / 'train_config.json'
CKPT = RUN_DIR / 'model_best.pt'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'shared'))

import torch
from shared.configs.schema import TrainConfig
from shared.train import _resolve_model_cls
from shared.eval_module import EvalModule

def main():
    t0 = time.time()
    rec = {
        'baseline_name': 'grid18_200ep_baseline',
        'run_dir': str(RUN_DIR),
        'config_path': str(CONFIG),
        'model_path': str(CKPT),
    }
    try:
        cfg = TrainConfig.model_validate(json.loads(CONFIG.read_text()))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_cls = _resolve_model_cls(cfg)
        model = model_cls(**cfg.arch_kwargs).to(device)
        state = torch.load(CKPT, map_location=device, weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        evaluator = EvalModule(cfg.split_manifest_path, cfg.data_dir)
        holdout = evaluator.evaluate(model, split='holdout', seed=cfg.seed, batch_size=4, device=device)
        val = {}
        metrics_path = RUN_DIR / 'metrics.json'
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            val = metrics.get('val_metrics') or {}
        rec.update({
            'status': 'ok',
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'seconds': time.time() - t0,
            'val_metrics': val,
            'holdout_metrics': holdout,
            'eval_hash': getattr(evaluator, 'eval_hash', ''),
            'split_hash': getattr(evaluator, 'split_hash', ''),
            'config': {
                'arch_name': cfg.arch_name,
                'epochs': cfg.epochs,
                'seed': cfg.seed,
                'batch_size': cfg.batch_size,
                'lr': cfg.lr,
                'loss_name': cfg.loss_name,
                'arch_kwargs': dict(cfg.arch_kwargs or {}),
            },
        })
        print('OK baseline holdout_r2_median', holdout.get('r2_median'), 'holdout_r2_global', holdout.get('r2_global'), 'holdout_mae_median', holdout.get('mae_median'), flush=True)
    except Exception as e:
        rec.update({'status': 'error', 'error': repr(e), 'traceback': traceback.format_exc(), 'seconds': time.time() - t0})
        print('ERR', repr(e), flush=True)
    out = WORK / 'grid18_200ep_baseline_holdout_eval.json'
    out.write_text(json.dumps(rec, indent=2))
    print('SAVED', out, flush=True)

if __name__ == '__main__':
    main()
