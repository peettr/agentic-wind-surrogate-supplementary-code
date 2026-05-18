# Benchmark200 live submission summary

Campaign: `v5_controller_auto10_001_benchmark200`
Remote root: `<GRID_HPC_SOURCE_ROOT>`
Submitted at: 2026-04-26 13:05-13:06 EDT

All 10 benchmark200 jobs were submitted live to HTCondor. Configs were materialized with `epochs=200` and `batch_size=16` after local validation (`Validated 10 train configs`).

| run_id | cluster | tier | source_run_id | immediate status |
|---|---:|---|---|---|
| r_auto10b_00_afno_benchmark200 | 19105 | h100_a100_l40s_12gb | r_auto10_00_afno_smoke20_repair4 | completed quickly with FAILED, OOM after fallback 16 -> 8 on L40S |
| r_auto10b_01_attention_mamba_benchmark200 | 19106 | h100_a100_l40s_12gb | r_auto10_01_attention_mamba_smoke20 | running |
| r_auto10b_02_cnn_deeponet_benchmark200 | 19107 | a40_rtx6k_16gb | r_auto10_02_cnn_deeponet_smoke20 | running |
| r_auto10b_03_convnext_v2_unet_benchmark200 | 19108 | a40_rtx6k_16gb | r_auto10_03_convnext_v2_unet_smoke20 | running |
| r_auto10b_04_dilated_fno_benchmark200 | 19109 | h100_a100_l40s_12gb | r_auto10_04_dilated_fno_smoke20 | running |
| r_auto10b_05_dilated_hrformer_benchmark200 | 19110 | a40_rtx6k_16gb | r_auto10_05_dilated_hrformer_smoke20_retry1 | running |
| r_auto10b_06_ffno_benchmark200 | 19111 | h100_a100_l40s_12gb | r_auto10_06_ffno_smoke20 | running |
| r_auto10b_07_fno_encoder_decoder_benchmark200 | 19112 | h100_a100_l40s_12gb | r_auto10_07_fno_encoder_decoder_smoke20 | idle at verification time |
| r_auto10b_08_hrdcn_benchmark200 | 19113 | a40_rtx6k_16gb | r_auto10_08_hrdcn_smoke20_retry1 | idle at verification time |
| r_auto10b_09_hrnet_benchmark200 | 19114 | a40_rtx6k_16gb | r_auto10_09_hrnet_smoke20 | idle at verification time |

AFNO immediate failure evidence:

```text
OOM with batch=16; falling back to 8
OOM even with batch=8; aborting this experiment
metrics.status=failed, epochs_trained=0, gpu=NVIDIA L40S
```
