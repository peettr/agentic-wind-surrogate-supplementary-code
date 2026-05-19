# Sequential Locked Files â€” ä»Ž V3 å¤åˆ¶ï¼Œä¸å¯ä¿®æ”¹

è¿™äº›æ–‡ä»¶æ˜¯ V3 è®­ç»ƒç®¡çº¿çš„ä¸€éƒ¨åˆ†ï¼ŒV4 çš„ codegen å’Œ workflow å¿…é¡»å…¼å®¹è¿™äº›æ–‡ä»¶çš„æŽ¥å£ã€‚

## é”å®šæ–‡ä»¶ï¼ˆä¸å¯ä¿®æ”¹ï¼‰

| æ–‡ä»¶ | è¯´æ˜Ž | CRC è·¯å¾„ |
|------|------|---------|
| `shared/train.py` | è®­ç»ƒä¸»è„šæœ¬ï¼Œè¯» train_config.jsonï¼Œå†™ metrics.json + FINISHED | `<HYBRID_HPC_SOURCE_ROOT>/shared/train.py` |
| `shared/losses.py` | Loss å‡½æ•°å®šä¹‰ï¼ˆmasked_l1, huber ç­‰ï¼‰ | åŒä¸Š |
| `shared/eval_module.py` | è¯„ä¼°é€»è¾‘ + hash æ ¡éªŒ | åŒä¸Š |
| `shared/data/split_manifest.json` | 20 seed çš„ train/val/holdout åˆ’åˆ† | åŒä¸Š |
| `shared/data/all_data.pt` | 4.1GBï¼Œ993 patchesï¼ˆCRC onlyï¼‰ | åŒä¸Š |

## å‚è€ƒæ–‡ä»¶ï¼ˆå¯ä¿®æ”¹ï¼‰

| æ–‡ä»¶ | è¯´æ˜Ž |
|------|------|
| `shared/models/*.py` | V3 çš„ 34 ä¸ªæ¨¡åž‹å®žçŽ°ï¼Œä½œä¸º codegen çš„å‚è€ƒèµ·ç‚¹ |

## Train.py åˆçº¦ï¼ˆcodegen å¿…é¡»éµå®ˆï¼‰

train.py æœŸæœ›çš„ `train_config.json` æ ¼å¼ï¼š
```json
{
  "experiment_id": "string",
  "arch_name": "string",
  "arch_kwargs": {"n_c": 16, "depth": 7, ...},
  "loss_name": "string",
  "loss_kwargs": {},
  "seed": 1,
  "epochs": 200,
  "lr": 0.001,
  "batch_size": 16,
  "checkpoint_interval": 50,
  "data_dir": "<HPC_PATH>/baseline_source/shared/data",
  "results_dir": "<HPC_PATH>/hybrid/campaigns/xxx/runs/exp_id",
  "split_manifest_path": "<HPC_PATH>/baseline_source/shared/data/split_manifest.json",
  "eval_splits": ["val"],
  "script_path": "optional codegen .py path"
}
```

train.py çš„è¾“å‡ºï¼š
- `$RESULTS_DIR/metrics.json` â€” åŒ…å« val_r2_median, val_r2_mean ç­‰
- `$RESULTS_DIR/FINISHED` â€” å®Œæˆæ ‡è®°
- `$RESULTS_DIR/FAILED` â€” å¤±è´¥æ ‡è®°
- `$RESULTS_DIR/checkpoint.pt` â€” æ¯ 50 epoch ä¿å­˜

## ç¦æ­¢ä¼ é€’çš„ V3 ä¿¡æ¯

- RÂ² å€¼ã€æŽ’åã€åˆ†çº§ï¼ˆTier A/B/C/Dï¼‰
- ä»»ä½•å®žéªŒç»“è®ºï¼ˆ"FNO ä¸å¦‚ UNet" ç­‰ï¼‰
- Sequential å¿…é¡»ä»Žé›¶å‘çŽ°ä¸€åˆ‡

## CRC æ“ä½œè§„åˆ™

- **æäº¤ CRC å·¥ä½œå‰å¿…é¡»è¯» `CRC-WORKFLOW.md`**
- ä¸åˆ  ControlMaster socketï¼Œä¸å°è¯• reconnect
- SSH è¶…æ—¶ â‰¥ 30 ç§’
- GPU è®­ç»ƒç”¨ external schedulerï¼ŒCPU ä»»åŠ¡ç”¨ SGE qsub
- wrapper è„šæœ¬å¿…é¡»å…ˆ `module load conda` + `conda activate graphwind`
- `initialdir` å¿…é¡»æ˜¯é¡¹ç›®æ ¹ç›®å½•ï¼ˆ`train.py` çš„ import éœ€è¦ `shared/` åœ¨ Python è·¯å¾„ä¸Šï¼‰






