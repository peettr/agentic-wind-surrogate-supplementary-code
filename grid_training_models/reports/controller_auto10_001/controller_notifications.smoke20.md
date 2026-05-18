- 2026-04-26T05:30:34.887991+00:00 `r_auto10_00_afno_smoke20` -> `NEEDS_DIAGNOSIS:COLLECT_MORE_EVIDENCE` action=`COLLECT_MORE_EVIDENCE`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_01_attention_mamba_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_02_cnn_deeponet_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_03_convnext_v2_unet_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_04_dilated_fno_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_05_dilated_hrformer_smoke20` -> `NEEDS_DIAGNOSIS:COLLECT_MORE_EVIDENCE` action=`COLLECT_MORE_EVIDENCE`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_06_ffno_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_07_fno_encoder_decoder_smoke20` -> `RUNNING:WAIT` action=`WAIT`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_08_hrdcn_smoke20` -> `NEEDS_DIAGNOSIS:COLLECT_MORE_EVIDENCE` action=`COLLECT_MORE_EVIDENCE`
- 2026-04-26T05:30:34.887991+00:00 `r_auto10_09_hrnet_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T05:46:21.544679+00:00 `r_auto10_07_fno_encoder_decoder_smoke20` -> `PASS:RECORD_RESULT` action=`RECORD_RESULT`
- 2026-04-26T11:51:55.947711+00:00 `r_auto10_00_afno_smoke20` -> `HIGH_VRAM:REPAIR` action=`REPAIR`
- 2026-04-26T11:51:55.947711+00:00 `r_auto10_05_dilated_hrformer_smoke20` -> `HIGH_VRAM:RETRY` action=`RETRY`
- 2026-04-26T11:51:55.947711+00:00 `r_auto10_08_hrdcn_smoke20` -> `HIGH_VRAM:RETRY` action=`RETRY`
- 2026-04-26T15:55:23+00:00 `r_auto10_00_afno_smoke20` -> `AUTO_FAIL_MAX_TOTAL_ATTEMPTS:AUTO_FAIL` action=`AUTO_FAIL` reason=`repair_count=4, total_attempts=5, repair4 low performance after memory fix`


## AFNO smoke policy correction: PASS after repair4
- timestamp: 2026-04-26T16:19:46.774505+00:00
- run_id: r_auto10_00_afno_smoke20
- current_run_id/source for benchmark200: r_auto10_00_afno_smoke20_repair4
- state_key: PASS:RECORD_RESULT
- policy: smoke20 is code/runtime validation only; low smoke R² is not a rejection criterion.
- evidence: FINISHED, metrics.status=ok, 20/20 epochs, model artifacts present, FAILED absent.


## Retry smoke results recorded as PASS
- timestamp: 2026-04-26T16:24:15.136035+00:00
- r_auto10_05_dilated_hrformer_smoke20: current_run_id=r_auto10_05_dilated_hrformer_smoke20_retry1, state_key=PASS:RECORD_RESULT, policy=code/runtime validation only.
- r_auto10_08_hrdcn_smoke20: current_run_id=r_auto10_08_hrdcn_smoke20_retry1, state_key=PASS:RECORD_RESULT, policy=code/runtime validation only.
