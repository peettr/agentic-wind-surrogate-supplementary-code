# V3 â†’ V4 Hard Constraints (A + D ç±»)

the human researcher ç¡®è®¤ï¼šA ç±»ï¼ˆç¡¬ä»¶/ç‰©ç†çº¦æŸï¼‰å’Œ D ç±»ï¼ˆå·¥ç¨‹çº¦æŸï¼‰å¯ä»¥åŠ å…¥ V4ã€‚
B ç±»ï¼ˆå®žéªŒå‘çŽ°ï¼‰å’Œ C ç±»ï¼ˆæž¶æž„ç»“è®ºï¼‰ä¸å¯åŠ å…¥â€”â€”Explorer å¿…é¡»è‡ªå·±å‘çŽ°ã€‚

---

## A ç±»ï¼šç¡¬ä»¶/ç‰©ç†çº¦æŸ

### A1. batch_size = 16ï¼ˆç¡¬é”å®šï¼‰
- **çº¦æŸ**ï¼š`batch_size` å›ºå®šä¸º 16ï¼Œä¸è¿› HP ç©ºé—´
- **fallback**ï¼štrain.py å†…éƒ¨ OOM æ—¶è‡ªåŠ¨é™ä¸º 8ï¼ˆè¿™æ˜¯ train.py çš„é€»è¾‘ï¼ŒV4 ä¸éœ€è¦ç®¡ï¼‰
- **åŽŸå› **ï¼šV3 æ‰€æœ‰æ­£å¼å®žéªŒå‡ç”¨ batch=16ï¼Œè¿™æ˜¯ç»è¿‡éªŒè¯çš„ç¨³å®šé…ç½®
- **å®žçŽ°**ï¼šHPSpace ä¸­ç§»é™¤ `batch_size` ç»´åº¦ï¼Œæˆ–å›ºå®šä¸º `[16]`

### A2. GPU VRAM åˆ†çº§çº¦æŸï¼ˆç¡¬çº¦æŸï¼‰
ä¸åŒæ¨¡åž‹ Ã— n_c ç»„åˆéœ€è¦ä¸åŒç­‰çº§çš„ GPUï¼š

| æ¨¡åž‹å‚æ•°é‡ | éœ€è¦ GPU VRAM | Condor requirements |
|-----------|:------------:|---------------------|
| < 35M (n_c=16, 7-level UNet) | â‰¥ 22GB | `GPUs_GlobalMemoryMb >= 22000` |
| 35-150M (n_c=32, 7-level UNet) | â‰¥ 40GB | `GPUs_GlobalMemoryMb >= 40000` |
| 150-500M (DilatedUNet n_c=32) | â‰¥ 45GB | `GPUs_GlobalMemoryMb >= 45000` |
| > 500M | ä¸å…è®¸ | ï¼ˆPeter æŒ‡å®šä¸Šé™ 150Mï¼Œä½† DilatedUNet 498M å·²åœ¨ V3 ä¸­è¿è¡Œï¼‰ |

**æ³¨æ„**ï¼šDilatedUNet depth=7 n_c=32 = 498M paramsï¼Œè™½ç„¶è¶…è¿‡ 150M ä¸Šé™ï¼Œä½† V3 ä¸­åœ¨ L40S(45GB) ä¸ŠæˆåŠŸè¿è¡Œã€‚V4 ä¸­ä¿ç•™æ­¤æ¨¡åž‹ã€‚

**å®žçŽ°**ï¼š`generate_v6_round.py` æ ¹æ® arch_name + n_c è‡ªåŠ¨è®¡ç®— paramsï¼Œè®¾ç½®å¯¹åº”çš„ Condor requirementsã€‚

### A3. activation å†…å­˜æ˜¯ OOM æ ¹å› ï¼ˆæŒ‡å¯¼åŽŸåˆ™ï¼‰
- **çº¦æŸ**ï¼šä¸åªæ˜¯å‚æ•°é‡ï¼Œactivation å†…å­˜ï¼ˆchÂ²Ã—HÂ²Ã—WÂ²Ã—depthÃ—batchï¼‰æ‰æ˜¯ OOM çš„çœŸæ­£åŽŸå› 
- **å½±å“**ï¼šé«˜åˆ†è¾¨çŽ‡ï¼ˆ640Ã—640ï¼‰+ å¤§ channelï¼ˆn_c=48ï¼‰+ æ·±ç½‘ç»œï¼ˆdepth=7ï¼‰ç»„åˆå¯èƒ½ OOMï¼Œå³ä½¿å‚æ•°é‡ < 150M
- **å®žçŽ°**ï¼š`generate_v6_round.py` ä¸­å¯¹ VRAM æ•æ„Ÿçš„é…ç½®ï¼ˆn_câ‰¥48 + depth=7ï¼‰è‡ªåŠ¨è¦æ±‚ â‰¥45GB GPU

### A4. stride-2 stem é™ä½Žåˆ†è¾¨çŽ‡ï¼ˆå·²çŸ¥ä»£ä»·ï¼‰
- **çº¦æŸ**ï¼šTier C+D æ¨¡åž‹ä½¿ç”¨ stride-2 stemï¼Œè¾“å…¥ä»Ž 640Ã—640 é™è‡³ 320Ã—320
- **å½±å“**ï¼šè¿™äº›æ¨¡åž‹çš„ RÂ² ä¼šå› åˆ†è¾¨çŽ‡é™ä½Žè€Œåä½Žï¼Œä¸æ˜¯æ¨¡åž‹æœ¬èº«å·®
- **V4 ä¸­**ï¼š640Ã—640 æ¨¡åž‹å’Œ 320Ã—320 æ¨¡åž‹ä¸åº”ç›´æŽ¥æ¯”è¾ƒ RÂ²
- **å®žçŽ°**ï¼šåœ¨ candidate_library çš„ notes ä¸­æ³¨æ˜Žå“ªäº›æ¨¡åž‹ç”¨ stride-2

### A5. SDF æ•°æ®å¿…é¡»é¢„ç”Ÿæˆ
- **çº¦æŸ**ï¼š`input_features=height_sdf_normal` éœ€è¦ `shared/sdf.py` é¢„ç”Ÿæˆçš„ 3ch .npy æ–‡ä»¶
- **æ•°æ®è·¯å¾„**ï¼š`<PROJECT_HPC_ROOT>/data/lu_sdf_640/`
- **å®žçŽ°**ï¼š`generate_v6_round.py` æ£€æŸ¥ SDF æ•°æ®ç›®å½•æ˜¯å¦å­˜åœ¨ï¼Œä¸å­˜åœ¨åˆ™è·³è¿‡ SDF input é€‰é¡¹

### A6. eval_module.py é”å®š
- **çº¦æŸ**ï¼ševal_module.py çš„ SHA-256 hash å†™å…¥æ¯ä¸ª metrics.json çš„ `eval_hash` å­—æ®µ
- **åŽŸå› **ï¼šä¿è¯æ‰€æœ‰å®žéªŒä½¿ç”¨ç›¸åŒçš„è¯„ä¼°æ–¹æ³•ï¼Œå¯å¯¹æ¯”
- **å®žçŽ°**ï¼šV4 campaign init é˜¶æ®µéªŒè¯ eval_hash ä¸Ž V3 ä¸€è‡´

---

## D ç±»ï¼šå·¥ç¨‹çº¦æŸï¼ˆCondor/æäº¤ï¼‰

### D1. Condor GPU regexp çº¦æŸï¼ˆç¡¬çº¦æŸï¼‰
- **æ­£ç¡®å†™æ³•**ï¼š
  ```
  requirements = (regexp("qa-h100-", Machine) || regexp("qa-a100-", Machine) || regexp("qa-l40s-", Machine))
  ```
- **ä¸èƒ½ç”¨**ï¼š`GPUs_GlobalMemoryMb`ï¼ˆCondor å°† undefined å±žæ€§è§†ä¸ºåŒ¹é…ï¼Œå¯¼è‡´åˆ†åˆ°æ—  GPU èŠ‚ç‚¹ï¼‰
- **GPU ç±»åž‹**ï¼š
  - H100: 80GB VRAM, ~60 TFLOPS FP32
  - L40S: 48GB VRAM, ~90 TFLOPS FP32
  - A40: 48GB VRAM, ~37 TFLOPS FP32
  - A6000: 48GB VRAM, ~39 TFLOPS FP32
  - RTX6000 Ada: 48GB VRAM, ~91 TFLOPS FP32
  - A10: 24GB VRAM, ~31 TFLOPS FP32
  - 2080 Ti: 11GB VRAM, ~13 TFLOPSï¼ˆå¤ªå°ï¼Œä¸ç”¨äºŽè®­ç»ƒï¼‰
- **å®žçŽ°**ï¼š`generate_v6_round.py` æ ¹æ®æ¨¡åž‹ VRAM éœ€æ±‚è‡ªåŠ¨ç”Ÿæˆ requirements

### D2. request_memory = 16GBï¼ˆç¡¬çº¦æŸï¼‰
- **è®­ç»ƒ**ï¼š`request_memory = 16 GB`ï¼ˆ8GB è¯„ä¼° OOMï¼‰
- **æ³¨æ„**ï¼š`request_memory = 64 GB` ä¼šè¿‡æ»¤æŽ‰æ‰€æœ‰ H100 slot
- **å®žçŽ°**ï¼šå›ºå®šä¸º 16GB

### D3. æŽ’é™¤ qa-a10-005ï¼ˆç¡¬çº¦æŸï¼‰
- **çº¦æŸ**ï¼š`requirements = && Machine != "<HPC_GPU_NODE>"`
- **åŽŸå› **ï¼šauto_v2 ä¸­åå¤å‡ºæƒé™/æ—¥å¿—å†™å…¥é—®é¢˜
- **å®žçŽ°**ï¼šæ‰€æœ‰ submit æ–‡ä»¶åŠ å…¥æ­¤æŽ’é™¤

### D4. Condor wrapper è„šæœ¬ï¼ˆå›ºå®šæ¨¡æ¿ï¼‰
```bash
#!/bin/bash
if [ -r /opt/crc/Modules/current/init/bash ]; then
  source /opt/crc/Modules/current/init/bash
fi
module load conda/25.9.1
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind
cd <BASELINE_HPC_SOURCE_ROOT>
```
- **æ³¨æ„**ï¼šV4 çš„ shared/ åœ¨ CRC ä¸Šä½äºŽ auto_v3/shared/ï¼ˆV4 å¤ç”¨ V3 çš„ shared/ï¼‰
- **å®žçŽ°**ï¼š`templates/condor_wrapper.sh` å›ºå®šä¸å˜

### D5. CRLF å¤„ç†ï¼ˆæ¯æ¬¡æäº¤å‰ï¼‰
- **çº¦æŸ**ï¼šæ‰€æœ‰ä»Ž Windows ä¸Šä¼ çš„ .sh/.json å¿…é¡» `tr -d '\r'`
- **å®žçŽ°**ï¼š`generate_v6_round.py` ç”Ÿæˆæ–‡ä»¶åŽï¼Œåœ¨ä¸Šä¼ å‰ç»Ÿä¸€å¤„ç† CRLF

### D6. SSH è¿žæŽ¥æ–¹å¼ï¼ˆå›ºå®šï¼‰
- **æ­£ç¡®**ï¼š`wsl bash -lc "ssh -o ControlPath=<SSH_CONTROL_PATH> lhu1@<HPC_FILE_LOGIN> '...'"`
- **è¶…æ—¶**ï¼šexec â‰¥ 60sï¼ŒConnectTimeout â‰¥ 30s
- **æ¢å¤**ï¼šPeter æ‰‹åŠ¨åœ¨ WSL ç»ˆç«¯é‡å»º ControlMaster socket

### D7. æ–‡ä»¶æƒé™ï¼ˆæ¯æ¬¡æäº¤å‰ï¼‰
- **çº¦æŸ**ï¼š`chmod 666 train.out train.err` é˜²æ­¢ Permission denied hold
- **å®žçŽ°**ï¼šsubmit è„šæœ¬ä¸­ç»Ÿä¸€å¤„ç†

### D8. CRC é¡¹ç›®è·¯å¾„ï¼ˆå›ºå®šï¼‰
- **é¡¹ç›®æ ¹**ï¼š`<BASELINE_HPC_SOURCE_ROOT>`
- **æ•°æ®**ï¼š`<PROJECT_HPC_ROOT>/data`
- **V4 ä»£ç **ï¼š`<BASELINE_HPC_SOURCE_ROOT>/`ï¼ˆV4 çš„ shared/models/engine/templates å¤ç”¨ V3 è·¯å¾„ï¼‰

### D9. H100 ä¸“å±žçº¦æŸ
- **Condor cgroup memory limit ä¸¥æ ¼**ï¼š`request_memory` å¿…é¡» â‰¥ 32GB ç³»ç»Ÿå†…å­˜ï¼ˆä¸åªæ˜¯ GPUï¼‰
- **H100 æ˜¯ condo èŠ‚ç‚¹**ï¼šbackfill æ¨¡å¼ï¼Œowner ä¸ç”¨æ—¶åŠ¨æ€åŠ å…¥ï¼Œå¯èƒ½è¢«æŠ¢å 
- **å®žçŽ°**ï¼šå¯¹ 498M+ å‚æ•°æ¨¡åž‹ï¼ˆDilatedUNetï¼‰ï¼Œrequest_memory=32GB

---

## æ±‡æ€»ï¼šV4 HP ç©ºé—´ï¼ˆåŠ å…¥ A+D çº¦æŸåŽï¼‰

### å›ºå®šå€¼ï¼ˆä¸è¿›æœç´¢ç©ºé—´ï¼‰
| ç»´åº¦ | å›ºå®šå€¼ | çº¦æŸç¼–å· |
|------|--------|---------|
| batch_size | 16 | A1 |
| epochs | æŒ‰é˜¶æ®µï¼š20/smoke, 200/focus, 1000/long | æµç¨‹çº¦æŸ |
| seed | 1ï¼ˆå¤š seed é˜¶æ®µé™¤å¤–ï¼‰ | æµç¨‹çº¦æŸ |

### æœç´¢ç©ºé—´
| ç»´åº¦ | å€¼ | çº¦æŸ |
|------|-----|------|
| arch_name | 34 ä¸ªæ¨¡åž‹ï¼ˆå…¨éƒ¨ï¼‰ | æ— æŽ’é™¤ |
| depth | 5, 6, 7 | â€” |
| n_c | 8, 16, 32, 48 | A2ï¼ˆè‡ªåŠ¨é€‰ GPUï¼‰ |
| loss_name | masked_l1, masked_l1_gradient, masked_huber | â€” |
| lr | 3e-4, 5e-4, 7e-4, 1e-3, 2e-3 | â€” |
| weight_decay | 0, 1e-5, 1e-4 | â€” |
| scheduler | none, cosine | â€” |
| gradient_clip | none, 0.5 | â€” |
| use_ema | false, true | â€” |
| ema_decay | 0.999 | â€” |
| augmentation | false, true | â€” |
| input_features | height, height_sdf, height_sdf_normal | A5ï¼ˆæ£€æŸ¥æ•°æ®å­˜åœ¨ï¼‰ |

### Condor è‡ªåŠ¨é…ç½®ï¼ˆæ ¹æ®æ¨¡åž‹ï¼‰
| æ¡ä»¶ | requirements | request_memory |
|------|-------------|----------------|
| params < 35M | GPUs_GlobalMemoryMb >= 22000 | 16 GB |
| params 35-150M | GPUs_GlobalMemoryMb >= 40000 | 16 GB |
| params > 150M | GPUs_GlobalMemoryMb >= 45000 | 32 GB |

æ‰€æœ‰ submit æ–‡ä»¶é™„åŠ ï¼š`&& Machine != "<HPC_GPU_NODE>"`ï¼ˆD3ï¼‰

