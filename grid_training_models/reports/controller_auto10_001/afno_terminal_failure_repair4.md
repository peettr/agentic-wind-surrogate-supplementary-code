# AFNO terminal decision after repair4

Decision time: 2026-04-26T11:55:23-04:00

## Final controller status

`r_auto10_00_afno_smoke20` is marked terminal fail:

```text
status = AUTO_FAIL_MAX_TOTAL_ATTEMPTS
secondary_status = LOW_PERFORMANCE_AFTER_REPAIR
memory_fixed = true
promotion_allowed = false
terminal = true
```

## Budget accounting

```text
max_retries = 3
max_repairs = 2
max_total_attempts = 5
retry_count = 0
repair_count = 4
total_attempts = 5
```

The submitted path was original smoke20 plus repair1, repair2, repair3, and repair4. Under the controller budget, no repair5 should be submitted automatically.

## Repair4 evidence

```text
run_id = r_auto10_00_afno_smoke20_repair4
cluster = 19092
GPU = NVIDIA L40S
params = 1,331,009
batch_size = 8
FINISHED = true
FAILED = false
metrics.status = ok
epochs_trained = 20
peak_vram_gb = 30.077748736
wall_time_sec = 1190.040122270584
val_r2_median = 0.07525065874468784
val_r2_global = 0.3314944664755364
val_mae_median = 0.1401417851448059
```

Repair4 fixed the L40S batch-8 OOM condition, but the validation R² is far below the v5 baseline reference from `auto_v3/campaigns/grid18_200ep/baseline` (`val_r2_median = 0.7085034105693335`). Therefore the run is not promotion-worthy and is terminal under the retry/repair budget.
