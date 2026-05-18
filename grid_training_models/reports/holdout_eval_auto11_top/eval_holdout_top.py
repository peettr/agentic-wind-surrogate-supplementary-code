#!/usr/bin/env python3
from __future__ import annotations
import json, sys, os, time, traceback
from pathlib import Path

ROOT = Path('<GRID_HPC_SOURCE_ROOT>')
WORK = Path('<GRID_HPC_SOURCE_ROOT>/reports/holdout_eval_auto11_top')
RUNS = json.loads((WORK/'selected_runs.json').read_text())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'shared'))

import torch
from shared.configs.schema import TrainConfig
from shared.train import _resolve_model_cls, _compute_sdf_features
from shared.eval_module import EvalModule

out=[]
for item in RUNS:
    t0=time.time()
    cfg_path=Path(item['config_path'])
    rec=dict(item)
    rec['config_path']=str(cfg_path)
    try:
        cfg=TrainConfig.model_validate(json.loads(cfg_path.read_text()))
        model_path=Path(cfg.results_dir)/'model_best.pt'
        rec['model_path']=str(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'missing model_best.pt: {model_path}')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_cls=_resolve_model_cls(cfg)
        model=model_cls(**cfg.arch_kwargs).to(device)
        state=torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state=state['model_state_dict']
        model.load_state_dict(state)
        evaluator=EvalModule(cfg.split_manifest_path, cfg.data_dir)
        if cfg.input_features != 'height':
            orig=evaluator._predict
            def _predict_with_sdf(model, X, batch_size, device):
                return orig(model, _compute_sdf_features(X, cfg.input_features), batch_size, device)
            evaluator._predict=_predict_with_sdf
        hold=evaluator.evaluate(model, split='holdout', seed=cfg.seed, batch_size=4, device=device)
        # load existing val metrics for side-by-side
        metrics_path=Path(cfg.results_dir)/'metrics.json'
        val={}
        if metrics_path.exists():
            m=json.loads(metrics_path.read_text())
            val=m.get('val_metrics') or {}
        rec.update({
            'status':'ok',
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'seconds': time.time()-t0,
            'holdout_metrics': hold,
            'val_metrics': val,
            'eval_hash': getattr(evaluator,'eval_hash',''),
            'split_hash': getattr(evaluator,'split_hash',''),
        })
        print('OK', rec['run_id'], 'holdout_r2_median', hold.get('r2_median'), 'holdout_r2_global', hold.get('r2_global'), flush=True)
    except Exception as e:
        rec.update({'status':'error','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.time()-t0})
        print('ERR', rec.get('run_id'), repr(e), flush=True)
    out.append(rec)
    (WORK/'holdout_results.partial.json').write_text(json.dumps(out, indent=2))
(WORK/'holdout_results.json').write_text(json.dumps(out, indent=2))
print('SAVED', WORK/'holdout_results.json')
