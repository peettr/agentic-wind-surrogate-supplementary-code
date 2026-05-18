# Auto V6 1000ep posthoc corrected-val55 checkpoint sweep summary

Selection rule: for each seed/run, choose the checkpoint with the highest V4 `stratified_v1` corrected val55 median R?; report the corresponding holdout47 and total102 metrics.

| Rank | Candidate | n | Mean val55 | Mean holdout47 | Mean total102 | Seed total102 values | Epochs |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | Top6 anisotropic_kernel_operator R11 nc20 lr4e-4 | 3 | 0.8017 | 0.7875 | **0.7946** | s2=0.779451;s3=0.836188;s4=0.768137 | s2:ep900;s3:ep400;s4:ep350 |
| 2 | Top3 anisotropic_kernel_operator R12 nc24 lr3.5e-4 | 3 | 0.7917 | 0.7796 | **0.7885** | s2=0.780363;s3=0.815811;s4=0.769178 | s2:ep650;s3:ep800;s4:ep550 |
| 3 | Top5 terrain_conditioned_local_attention R22 retry1 | 3 | 0.7835 | 0.7666 | **0.7878** | s2=0.777564;s3=0.803392;s4=0.782581 | s2:ep900;s3:ep700;s4:ep350 |
| 4 | Top2 terrain_conditioned_local_attention R23 retry2 | 3 | 0.7866 | 0.7717 | **0.7839** | s2=0.775907;s3=0.792898;s4=0.782842 | s2:ep950;s3:ep800;s4:ep300 |
| 5 | Top1 boundary_gated_multiscale R12 nc24 lr3e-4 | 3 | 0.7703 | 0.7764 | **0.7809** | s2=0.763349;s3=0.810833;s4=0.768462 | s2:ep1000;s3:ep650;s4:ep400 |
| 6 | Top4 terrain_conditioned_local_attention R20 | 3 | 0.7803 | 0.7708 | **0.7789** | s2=0.760983;s3=0.798377;s4=0.777235 | s2:ep150;s3:ep500;s4:ep200 |
| 7 | Top7 boundary_gated_multiscale R08 lr4e-4 | 3 | 0.7767 | 0.7653 | **0.7761** | s2=0.768992;s3=0.794172;s4=0.765106 | s2:ep450;s3:ep300;s4:ep700 |
