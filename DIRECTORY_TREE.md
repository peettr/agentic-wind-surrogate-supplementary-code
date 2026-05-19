# Directory tree

```text
supplementary_code_release/
|-- documentation/
|   |-- cleaning_report.md
|   |-- code_inventory.md
|   |-- compute_environment.md
|   |-- data_format.md
|   |-- known_limitations.md
|   |-- metric_definitions.md
|   |-- split_and_selection_protocol.md
|   `-- workflow_description.md
|-- grid_training_models/
|   |-- metadata/
|   |   |-- grid_architectures.json
|   |   |-- grid_hp_candidates.json
|   |   `-- GRID_PROTOCOL.md
|   |-- model_rounds/
|   |   |-- round_001/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno/
|   |   |   |-- model_01_attention_mamba/
|   |   |   |-- model_02_cnn_deeponet/
|   |   |   |-- model_03_convnext_v2_unet/
|   |   |   |-- model_04_dilated_fno/
|   |   |   |-- model_05_dilated_hrformer/
|   |   |   |-- model_06_dilated_unet/
|   |   |   |-- model_07_ffno/
|   |   |   |-- model_08_fno_encoder_decoder/
|   |   |   `-- model_09_hrdcn/
|   |   |-- round_002/
|   |   |   |-- configs/
|   |   |   |-- model_00_hrformer/
|   |   |   |-- model_01_hrnet/
|   |   |   |-- model_02_mamba2d/
|   |   |   |-- model_03_mamba_attention/
|   |   |   |-- model_04_nafnet/
|   |   |   |-- model_05_perceiver_io/
|   |   |   |-- model_06_quadmamba/
|   |   |   |-- model_07_residual_spectral/
|   |   |   |-- model_08_sac_mamba/
|   |   |   `-- model_09_swin_unetr/
|   |   |-- round_003/
|   |   |   |-- configs/
|   |   |   |-- model_00_transolver/
|   |   |   |-- model_01_transolver_lite/
|   |   |   |-- model_02_ufno/
|   |   |   |-- model_03_umamba/
|   |   |   |-- model_04_unet_afno/
|   |   |   |-- model_05_unet_sdf_7level/
|   |   |   |-- model_06_unet_v3/
|   |   |   |-- model_07_uno/
|   |   |   |-- model_08_afno/
|   |   |   `-- model_09_attention_gate_unet/
|   |   |-- round_004/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno/
|   |   |   |-- model_01_attention_gate_unet/
|   |   |   |-- model_02_attention_mamba/
|   |   |   |-- model_03_cbam_unet/
|   |   |   |-- model_04_cnn_deeponet/
|   |   |   |-- model_05_cno_v2/
|   |   |   |-- model_06_convnext_v2_unet/
|   |   |   |-- model_07_dcn_unet/
|   |   |   |-- model_08_dilated_fno/
|   |   |   `-- model_09_dilated_hrformer/
|   |   |-- round_005/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno/
|   |   |   |-- model_01_attention_gate_unet/
|   |   |   |-- model_02_attention_mamba/
|   |   |   |-- model_03_cbam_unet/
|   |   |   |-- model_04_cnn_deeponet/
|   |   |   |-- model_05_cno_v2/
|   |   |   |-- model_06_convnext_v2_unet/
|   |   |   |-- model_07_dcn_unet/
|   |   |   |-- model_08_dilated_fno/
|   |   |   `-- model_09_dilated_hrformer/
|   |   |-- round_006/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno/
|   |   |   |-- model_01_attention_gate_unet/
|   |   |   |-- model_02_attention_mamba/
|   |   |   |-- model_03_cbam_unet/
|   |   |   |-- model_04_cnn_deeponet/
|   |   |   |-- model_05_cno_v2/
|   |   |   |-- model_06_convnext_v2_unet/
|   |   |   |-- model_07_dcn_unet/
|   |   |   |-- model_08_dilated_fno/
|   |   |   `-- model_09_dilated_hrformer/
|   |   |-- round_007/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_attention_mamba/
|   |   |   |-- model_02_cbam_unet/
|   |   |   |-- model_03_cnn_deeponet/
|   |   |   |-- model_04_cno_v2/
|   |   |   |-- model_05_convnext_v2_unet/
|   |   |   |-- model_06_dcn_unet/
|   |   |   |-- model_07_dilated_fno/
|   |   |   |-- model_08_dilated_hrformer/
|   |   |   `-- model_09_dilated_unet/
|   |   |-- round_008/
|   |   |   |-- configs/
|   |   |   |-- model_00_cbam_unet/
|   |   |   |-- model_01_cno_v2/
|   |   |   |-- model_02_dcn_unet/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_ffno/
|   |   |   |-- model_05_fno2d/
|   |   |   |-- model_06_fno_encoder_decoder/
|   |   |   |-- model_07_fourier_unet/
|   |   |   |-- model_08_hrdcn/
|   |   |   `-- model_09_hrformer/
|   |   |-- round_009/
|   |   |   |-- configs/
|   |   |   |-- model_00_dilated_unet/
|   |   |   |-- model_01_ffno/
|   |   |   |-- model_02_fno2d/
|   |   |   |-- model_03_fno_encoder_decoder/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_hrdcn/
|   |   |   |-- model_06_hrformer/
|   |   |   |-- model_07_hrnet/
|   |   |   |-- model_08_kan_unet/
|   |   |   `-- model_09_mamba2d/
|   |   |-- round_010/
|   |   |   |-- configs/
|   |   |   |-- model_00_dilated_unet/
|   |   |   |-- model_01_ffno/
|   |   |   |-- model_02_fno2d/
|   |   |   |-- model_03_fno_encoder_decoder/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_hrdcn/
|   |   |   |-- model_06_hrformer/
|   |   |   |-- model_07_hrnet/
|   |   |   |-- model_08_kan_unet/
|   |   |   `-- model_09_mamba2d/
|   |   |-- round_011/
|   |   |   |-- configs/
|   |   |   |-- model_00_ffno/
|   |   |   |-- model_01_fno2d/
|   |   |   |-- model_02_fno_encoder_decoder/
|   |   |   |-- model_03_fourier_unet/
|   |   |   |-- model_04_hrdcn/
|   |   |   |-- model_05_hrformer/
|   |   |   |-- model_06_hrnet/
|   |   |   |-- model_07_kan_unet/
|   |   |   |-- model_08_mamba2d/
|   |   |   `-- model_09_mamba_attention/
|   |   |-- round_012/
|   |   |   |-- configs/
|   |   |   |-- model_00_fno2d/
|   |   |   |-- model_01_fourier_unet/
|   |   |   |-- model_02_hrnet/
|   |   |   |-- model_03_kan_unet/
|   |   |   |-- model_04_mamba2d/
|   |   |   |-- model_05_mamba_attention/
|   |   |   |-- model_06_multiscale_conv/
|   |   |   |-- model_07_nafnet/
|   |   |   |-- model_08_perceiver_io/
|   |   |   `-- model_09_quadmamba/
|   |   |-- round_013/
|   |   |   |-- configs/
|   |   |   |-- model_00_kan_unet/
|   |   |   |-- model_01_mamba_attention/
|   |   |   |-- model_02_multiscale_conv/
|   |   |   |-- model_03_nafnet/
|   |   |   |-- model_04_perceiver_io/
|   |   |   |-- model_05_quadmamba/
|   |   |   |-- model_06_residual_spectral/
|   |   |   |-- model_07_sac_mamba/
|   |   |   |-- model_08_sac_unet/
|   |   |   `-- model_09_swin_unetr/
|   |   |-- round_014/
|   |   |   |-- configs/
|   |   |   |-- model_00_mamba_attention/
|   |   |   |-- model_01_multiscale_conv/
|   |   |   |-- model_02_nafnet/
|   |   |   |-- model_03_perceiver_io/
|   |   |   |-- model_04_quadmamba/
|   |   |   |-- model_05_residual_spectral/
|   |   |   |-- model_06_sac_mamba/
|   |   |   |-- model_07_sac_unet/
|   |   |   |-- model_08_swin_unetr/
|   |   |   `-- model_09_transolver/
|   |   |-- round_015/
|   |   |   |-- configs/
|   |   |   |-- model_00_multiscale_conv/
|   |   |   |-- model_01_perceiver_io/
|   |   |   |-- model_02_quadmamba/
|   |   |   |-- model_03_residual_spectral/
|   |   |   |-- model_04_sac_mamba/
|   |   |   |-- model_05_sac_unet/
|   |   |   |-- model_06_swin_unetr/
|   |   |   |-- model_07_transolver/
|   |   |   |-- model_08_transolver_lite/
|   |   |   `-- model_09_ufno/
|   |   |-- round_016/
|   |   |   |-- configs/
|   |   |   |-- model_00_multiscale_conv/
|   |   |   |-- model_01_residual_spectral/
|   |   |   |-- model_02_sac_mamba/
|   |   |   |-- model_03_sac_unet/
|   |   |   |-- model_04_swin_unetr/
|   |   |   |-- model_05_transolver/
|   |   |   |-- model_06_transolver_lite/
|   |   |   |-- model_07_ufno/
|   |   |   |-- model_08_umamba/
|   |   |   `-- model_09_unet_afno/
|   |   |-- round_017/
|   |   |   |-- configs/
|   |   |   |-- model_00_sac_unet/
|   |   |   |-- model_01_transolver/
|   |   |   |-- model_02_transolver_lite/
|   |   |   |-- model_03_ufno/
|   |   |   |-- model_04_umamba/
|   |   |   |-- model_05_unet_afno/
|   |   |   |-- model_06_unet_sdf_7level/
|   |   |   |-- model_07_unet_v2_reference_gradient_lr1e3_ema_cosine/
|   |   |   |-- model_08_unet_v3/
|   |   |   `-- model_09_uno/
|   |   |-- round_018/
|   |   |   |-- configs/
|   |   |   |-- model_00_transolver_lite/
|   |   |   |-- model_01_ufno/
|   |   |   |-- model_02_umamba/
|   |   |   |-- model_03_unet_afno/
|   |   |   |-- model_04_unet_sdf_7level/
|   |   |   |-- model_05_unet_v2_reference_gradient_lr1e3_ema_nosched/
|   |   |   |-- model_06_unet_v3/
|   |   |   |-- model_07_uno/
|   |   |   |-- model_08_umamba/
|   |   |   `-- model_09_unet_afno/
|   |   |-- round_019/
|   |   |   |-- configs/
|   |   |   |-- model_00_unet_sdf_7level/
|   |   |   |-- model_01_unet_v2_reference_gradient_lr1e3_noema_cosine/
|   |   |   |-- model_02_unet_v3/
|   |   |   |-- model_03_uno/
|   |   |   |-- model_04_unet_sdf_7level/
|   |   |   |-- model_05_unet_v2_reference_gradient_lr1e3_noema_nosched/
|   |   |   |-- model_06_unet_v2_reference_gradient_lr5e4_ema_cosine/
|   |   |   |-- model_07_unet_v2_reference_gradient_lr5e4_ema_nosched/
|   |   |   |-- model_08_unet_v2_reference_gradient_lr5e4_noema_cosine/
|   |   |   `-- model_09_unet_v2_reference_gradient_lr5e4_noema_nosched/
|   |   `-- round_020/
|   |       |-- configs/
|   |       |-- model_00_unet_v2_reference_l1_lr1e3_ema_cosine/
|   |       |-- model_01_unet_v3/
|   |       |-- model_02_uno/
|   |       |-- model_03_unet_v2_reference_l1_lr1e3_ema_nosched/
|   |       |-- model_04_unet_v2_reference_l1_lr1e3_noema_cosine/
|   |       |-- model_05_unet_v2_reference_l1_lr1e3_noema_nosched/
|   |       |-- model_06_unet_v2_reference_l1_lr5e4_ema_cosine/
|   |       |-- model_07_unet_v2_reference_l1_lr5e4_ema_nosched/
|   |       |-- model_08_unet_v2_reference_l1_lr5e4_noema_cosine/
|   |       `-- model_09_unet_v2_reference_l1_lr5e4_noema_nosched/
|   |-- shared/
|   |   |-- configs/
|   |   |   |-- __init__.py
|   |   |   |-- problem_definition.yaml
|   |   |   |-- schema.py
|   |   |   `-- search_space.json
|   |   |-- eval_module.py
|   |   |-- losses.py
|   |   |-- preflight.py
|   |   |-- preprocess_sdf.py
|   |   |-- sdf.py
|   |   |-- train.py
|   |   `-- train_refiner.py
|   |-- templates/
|   |   |-- condor_submit.template
|   |   `-- condor_wrapper.sh
|   |-- grid_all_metrics_summary.csv
|   |-- grid_top50_metrics_summary.csv
|   |-- holdout_results.json
|   |-- model_rounds_manifest.csv
|   |-- PACKAGE_MANIFEST.json
|   `-- README.md
|-- hybrid_training_models/
|   |-- configs/
|   |   |-- candidate_library.json
|   |   |-- explorer_config.json
|   |   |-- model_specs.yaml
|   |   |-- search_space.json
|   |   `-- suggestion_round1.json
|   |-- model_rounds/
|   |   |-- r000/
|   |   |   |-- r000_full_cno_lrbasis_nc16_lr0.0005_masked_l1_gradient/
|   |   |   |-- r000_full_dilated_unet_nc16_lr0.001_masked_l1_gradient/
|   |   |   |-- r000_full_fno2d_lora_nc16_lr0.0005_masked_l1_gradient/
|   |   |   |-- r000_full_fourier_unet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r000_full_fourier_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r000_full_fourier_unet_wtconv_nc16_lr0.0005_masked_l1/
|   |   |   |-- r000_full_unet_sdf_7level_nc32_lr0.0005_masked_l1/
|   |   |   |-- r000_full_unet_sdf_7level_nc32_lr0.001_masked_l1_gradient/
|   |   |   |-- r000_full_unet_v3_gausres_nc16_lr0.0005_masked_l1/
|   |   |   |-- r000_full_unet_v3_geomtok_nc16_lr0.0005_masked_l1_gradient/
|   |   |   `-- r000_full_unet_v3_pointrefine_nc16_lr0.0005_masked_l1_gradient/
|   |   |-- r001/
|   |   |   |-- r001_full_boundary_gated_multiscale_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r001_full_coord_field_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r001_full_fgmoe_decoder_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r001_full_fourier_unet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r001_full_multigrid_spectral_unet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r001_full_roughness_gated_ssm_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r001_full_separable_fourier_unet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r001_full_unet_sdf_7level_nc32_lr0.0005_masked_l1_gradient/
|   |   |   |-- r001_full_unet_sdf_7level_nc32_lr0.0005_masked_l1_gradient_retry1/
|   |   |   |-- r001_full_unet_v3_gausres_nc16_lr0.0005_masked_l1/
|   |   |   |-- r001_full_unet_v3_gausres_nc24_lr0.0005_masked_l1/
|   |   |   |-- r001_full_unet_v3_geomtok_nc16_lr0.0005_masked_l1/
|   |   |   `-- r001_full_unet_v3_pointrefine_nc16_lr0.0005_masked_l1/
|   |   |-- r002/
|   |   |   |-- r002_full_adaptive_afno_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r002_full_attn_ssm_neighbor_unet_nc16_lr0.0003_masked_l1/
|   |   |   |-- r002_full_boundary_gated_multiscale_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r002_full_boundary_token_film_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r002_full_cnn_deeponet_lowrank_nc16_lr0.0005_masked_l1/
|   |   |   |-- r002_full_crossaxis_attn_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r002_full_hypernet_adapter_unet_nc16_lr0.0003_masked_l1/
|   |   |   |-- r002_full_separable_fourier_unet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r002_full_unet_sdf_7level_nc24_lr0.0005_masked_l1/
|   |   |   |-- r002_full_unet_sdf_7level_nc32_lr0.0005_masked_l1_gradient/
|   |   |   |-- r002_full_unet_sdf_7level_nc32_lr0.0005_masked_l1_gradient_retry3/
|   |   |   |-- r002_full_unet_v3_gausres_nc32_lr0.0005_masked_l1_gradient/
|   |   |   `-- r002_full_unet_v3_geomtok_nc24_lr0.0005_masked_l1/
|   |   |-- r003/
|   |   |   |-- r003_full_attn_ssm_skip_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_boundary_gated_multiscale_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r003_full_boundary_gated_multiscale_unet_nc32_lr0.0005_masked_l1_gradient/
|   |   |   |-- r003_full_boundary_token_mixer_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_geo_gate_conv_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_liif_head_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_mean_residual_corrector_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_reduced_kv_context_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r003_full_separable_fourier_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r003_full_unet_sdf_7level_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r003_full_unet_v3_gausres_nc24_lr0.0005_masked_l1_gradient/
|   |   |   `-- r003_full_unet_v3_geomtok_nc24_lr0.0005_masked_l1_gradient/
|   |   |-- r004/
|   |   |   |-- r004_full_boundary_gated_multiscale_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r004_full_coord_field_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r004_full_derived_feature_aggregate_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r004_full_hrnet_nc24_lr0.0005_masked_l1/
|   |   |   |-- r004_full_hydra_lora_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r004_full_mean_residual_corrector_unet_nc24_lr0.0005_masked_l1_gradient/
|   |   |   |-- r004_full_tiled_spectral_mixer_unet_nc16_lr0.0005_masked_l1/
|   |   |   |-- r004_full_transolver_lite_nc24_lr0.0005_masked_l1/
|   |   |   |-- r004_full_unet_sdf_7level_nc24_lr0.0005_masked_l1_gradient/
|   |   |   `-- r004_full_unet_v3_geomtok_nc24_lr0.0005_masked_l1_gradient/
|   |   |-- r005/
|   |   |   |-- r005_full_beno_multitoken_film_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_grad/
|   |   |   |-- r005_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00035_mask/
|   |   |   |-- r005_full_boundary_gated_multiscale_unet_nc28_d6_height_sdf_normal_flip_rot_lr0.0005_maske/
|   |   |   |-- r005_full_coarse_grid_mp_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradient/
|   |   |   |-- r005_full_coarse_prior_adain_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradie/
|   |   |   |-- r005_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1/
|   |   |   |-- r005_full_mean_residual_corrector_unet_nc28_d6_height_sdf_normal_none_lr0.0005_masked_l1_g/
|   |   |   |-- r005_full_msca_unet_nc16_d6_height_none_lr0.0005_masked_l1/
|   |   |   |-- r005_full_squeeze_axial_detail_unet_nc16_d6_height_none_lr0.0005_masked_l1/
|   |   |   |-- r005_full_unet_sdf_7level_nc24_d7_height_sdf_normal_none_lr0.0005_masked_l1_gradient/
|   |   |   `-- r005_full_unet_v3_geomtok_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradient/
|   |   |-- r006/
|   |   |   |-- r006_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00045_mask/
|   |   |   |-- r006_full_boundary_gated_multiscale_unet_nc28_d6_height_sdf_normal_flip_rot_lr0.00035_mask/
|   |   |   |-- r006_full_boundary_strip_graph_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l/
|   |   |   |-- r006_full_dct_spectral_unet_nc24_d4_height_none_lr0.0005_masked_l1/
|   |   |   |-- r006_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_maske/
|   |   |   |-- r006_full_edge_offset_corrector_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gra/
|   |   |   |-- r006_full_hilo_attn_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradient/
|   |   |   |-- r006_full_hilo_attn_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradient_retry2/
|   |   |   |-- r006_full_iterative_residual_polisher_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked/
|   |   |   |-- r006_full_pid_dcb_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1_gradient/
|   |   |   |-- r006_full_pid_dcb_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1_gradient_retr/
|   |   |   |-- r006_full_unet_sdf_7level_nc28_d7_height_sdf_normal_none_lr0.0005_masked_l1/
|   |   |   |-- r006_full_unet_sdf_7level_nc32_d7_height_sdf_normal_none_lr0.0005_masked_l1/
|   |   |   |-- r006_full_unet_sdf_7level_nc32_d7_height_sdf_normal_none_lr0.0005_masked_l1_retry2/
|   |   |   `-- r006_full_unet_v3_geomtok_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1_gradient/
|   |   |-- r007/
|   |   |   |-- r007_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00045_mask/
|   |   |   |-- r007_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0004_maske/
|   |   |   |-- r007_full_cross_scale_operator_token_unet_nc16_d6_height_sdf_normal_flip_rot_lr0.0005_mask/
|   |   |   |-- r007_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00045_mask/
|   |   |   |-- r007_full_iterative_residual_polisher_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_mas/
|   |   |   |-- r007_full_lowrank_spatial_operator_unet_nc16_d6_height_sdf_normal_none_lr0.0005_masked_l1/
|   |   |   |-- r007_full_mean_residual_corrector_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00045_masked/
|   |   |   |-- r007_full_multiscale_linear_attention_decoder_unet_nc16_d6_height_sdf_normal_none_lr0.0005/
|   |   |   |-- r007_full_semi_lagrangian_warp_unet_nc16_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1/
|   |   |   |-- r007_full_sparse_residual_refinement_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_mask/
|   |   |   |-- r007_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1_gr/
|   |   |   `-- r007_full_unet_sdf_7level_nc24_d7_height_sdf_normal_none_lr0.00035_masked_l1_gradient/
|   |   |-- r008/
|   |   |   |-- r008_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r008_full_coarse_to_fine_ladder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1/
|   |   |   |-- r008_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_maske/
|   |   |   |-- r008_full_dual_branch_boundary_film_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_maske/
|   |   |   |-- r008_full_hyperinr_residual_head_unet_nc24_d6_height_sdf_normal_none_lr0.00035_masked_l1_g/
|   |   |   |-- r008_full_iterative_residual_polisher_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked/
|   |   |   |-- r008_full_multiscale_ssm_pyramid_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l/
|   |   |   |-- r008_full_noncausal_neighbor_ssm_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l/
|   |   |   |-- r008_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1_gr/
|   |   |   |-- r008_full_tt_adapter_decoder_unet_nc24_d7_height_sdf_normal_none_lr0.00035_masked_l1_gradi/
|   |   |   |-- r008_full_unet_sdf_7level_nc24_d7_height_sdf_normal_none_lr0.0003_masked_l1_gradient/
|   |   |   `-- r008_full_wavelet_residual_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked/
|   |   |-- r009/
|   |   |   |-- r009_full_boundary_gated_multiscale_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r009_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.000375_masked/
|   |   |   |-- r009_full_boundary_pressure_basis_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1_g/
|   |   |   |-- r009_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r009_full_dual_branch_boundary_film_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r009_full_dual_spatial_channel_agg_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r009_full_latent_grid_adapter_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l1_grad/
|   |   |   |-- r009_full_mlp_multiscale_decoder_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l1_g/
|   |   |   |-- r009_full_physics_state_slice_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l/
|   |   |   |-- r009_full_spatial_prior_adapter_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l1_gr/
|   |   |   |-- r009_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00045_masked_l1_g/
|   |   |   `-- r009_full_unet_sdf_7level_nc24_d7_height_sdf_normal_none_lr0.00035_masked_l1/
|   |   |-- r010/
|   |   |   |-- r010_full_boundary_alltoall_surface_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.0004_mas/
|   |   |   |-- r010_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.00035_masked_l/
|   |   |   |-- r010_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r010_full_boundary_wtconv_large_receptive_unet_nc24_d6_height_sdf_normal_none_lr0.0004_mas/
|   |   |   |-- r010_full_cross_shape_decoder_adapter_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r010_full_dual_spatial_channel_agg_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l1/
|   |   |   |-- r010_full_geometry_landmark_bottleneck_operator_unet_nc24_d6_height_sdf_normal_none_lr0.00/
|   |   |   |-- r010_full_geometry_landmark_operator_token_unet_nc24_d6_height_sdf_normal_none_lr0.0004_ma/
|   |   |   |-- r010_full_physics_state_slice_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r010_full_pseudo_station_bottleneck_fusion_unet_nc24_d6_height_sdf_normal_none_lr0.0004_ma/
|   |   |   |-- r010_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0004_masked_l1_gr/
|   |   |   `-- r010_full_visual_context_replay_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1_gra/
|   |   |-- r011/
|   |   |   |-- r011_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r011_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.00035_mask/
|   |   |   |-- r011_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0003_masked_l1/
|   |   |   |-- r011_full_derived_feature_aggregate_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_maske/
|   |   |   |-- r011_full_dual_spatial_channel_agg_unet_nc24_d6_height_sdf_normal_none_lr0.0005_masked_l1/
|   |   |   |-- r011_full_dynamic_roughness_dispatch_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l/
|   |   |   |-- r011_full_geometry_modulated_afno_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l1_g/
|   |   |   |-- r011_full_laplace_rational_filter_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l1_g/
|   |   |   |-- r011_full_physics_state_slice_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked_l/
|   |   |   |-- r011_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0005_masked_l1_gr/
|   |   |   `-- r011_full_vmt_lite_shared_projection_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l/
|   |   |-- r012/
|   |   |   |-- r012_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.00035_masked/
|   |   |   |-- r012_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.00045_masked/
|   |   |   |-- r012_full_anisotropic_kernel_operator_unet_nc24_d6_height_sdf_normal_none_lr0.00035_masked/
|   |   |   |-- r012_full_band_adaptive_spectral_adapter_unet_nc20_d6_height_sdf_normal_none_lr0.0004_mask/
|   |   |   |-- r012_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0003_masked_l1/
|   |   |   |-- r012_full_boundary_ring_mixer_unet_nc24_d6_height_sdf_normal_none_lr0.00035_masked_l1_grad/
|   |   |   |-- r012_full_dynamic_roughness_dispatch_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l/
|   |   |   |-- r012_full_incremental_mode_gate_afno_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l/
|   |   |   |-- r012_full_laplace_rational_filter_unet_nc20_d6_height_sdf_normal_none_lr0.00035_masked_l1/
|   |   |   |-- r012_full_shared_norm_cross_shape_adapter_unet_nc24_d6_height_sdf_normal_none_lr0.00035_ma/
|   |   |   |-- r012_full_sparse_shared_projection_expert_adapter_unet_nc20_d5_height_sdf_normal_none_lr0/
|   |   |   `-- r012_full_texture_energy_modulated_ssm_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |-- r013/
|   |   |   |-- r013_full_anisotropic_boundary_hybrid_unet_nc20_d6_height_sdf_none_lr0.0004_masked_l1_grad/
|   |   |   |-- r013_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r013_full_anisotropic_kernel_operator_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r013_full_boundary_gated_multiscale_unet_nc20_d6_height_sdf_normal_none_lr0.0003_masked_l1/
|   |   |   |-- r013_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r013_full_coarse_interp_residual_head_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r013_full_heterogeneous_micro_adapter_mixture_unet_nc16_d6_height_sdf_normal_none_lr0.0004/
|   |   |   |-- r013_full_laplace_rational_filter_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1_g/
|   |   |   |-- r013_full_one_sided_lora_output_adapter_unet_nc24_d6_height_sdf_normal_none_lr0.0004_maske/
|   |   |   |-- r013_full_planar_multigrid_vcycle_unet_nc24_d7_height_sdf_normal_none_lr0.0004_masked_l1_g/
|   |   |   |-- r013_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_l1_gradie/
|   |   |   `-- r013_full_vq_pressure_regime_codebook_unet_nc16_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |-- r014/
|   |   |   |-- r014_full_anisotropic_boundary_hybrid_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r014_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r014_full_anisotropic_kernel_operator_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked/
|   |   |   |-- r014_full_anisotropic_vcycle_fusion_unet_nc24_d7_height_sdf_normal_none_lr0.0004_masked_l1/
|   |   |   |-- r014_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0003_masked_l1/
|   |   |   |-- r014_full_channel_transposed_attention_head_unet_nc20_d6_height_sdf_normal_none_lr0.0004_m/
|   |   |   |-- r014_full_colora_decoder_adapter_unet_nc20_d6_height_none_lr0.0004_masked_l1_gradient/
|   |   |   |-- r014_full_height_only_anisotropic_compact_unet_nc24_d6_height_none_lr0.0004_masked_l1_grad/
|   |   |   |-- r014_full_localized_integral_differential_unet_nc20_d6_height_none_lr0.0004_masked_l1_grad/
|   |   |   |-- r014_full_overlap_add_spectral_adapter_unet_nc20_d6_height_sdf_normal_none_lr0.00035_maske/
|   |   |   |-- r014_full_planar_multigrid_vcycle_unet_nc24_d7_height_sdf_normal_none_lr0.00035_masked_l1/
|   |   |   `-- r014_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0004_masked_l1_gr/
|   |   |-- r015/
|   |   |   |-- r015_full_allaround_strip_mixer_unet_nc24_d5_height_none_lr0.0004_masked_l1/
|   |   |   |-- r015_full_anisotropic_boundary_hybrid_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r015_full_anisotropic_kernel_operator_unet_nc24_d6_height_sdf_normal_none_lr0.00045_masked/
|   |   |   |-- r015_full_boundary_gated_multiscale_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked_hu/
|   |   |   |-- r015_full_boundary_path_message_unet_nc24_d5_height_sdf_normal_none_lr0.0004_masked_l1_gra/
|   |   |   |-- r015_full_frame_evolution_conditioner_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0004_mas/
|   |   |   |-- r015_full_fusion_basis_decoder_unet_nc20_d5_height_sdf_normal_none_lr0.0004_masked_l1_grad/
|   |   |   |-- r015_full_haloed_local_fourier_unet_nc24_d5_height_sdf_normal_none_lr0.0004_masked_l1_grad/
|   |   |   |-- r015_full_localized_integral_differential_unet_nc24_d6_height_sdf_normal_none_lr0.0004_mas/
|   |   |   |-- r015_full_multi_scale_fourier_basis_head_unet_nc24_d6_height_sdf_none_lr0.0004_masked_l1_g/
|   |   |   |-- r015_full_planar_multigrid_vcycle_unet_nc24_d6_height_sdf_none_lr0.0004_masked_l1_gradient/
|   |   |   `-- r015_full_tt_adapter_decoder_unet_nc24_d6_height_sdf_normal_flip_rot_lr0.0004_masked_l1_gr/
|   |   |-- r016/
|   |   |   |-- r016_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0/
|   |   |   |-- r016_full_boundary_path_message_unet_nc24_d6_height_sdf_normal_light_geometric_lr0.0004_ma/
|   |   |   |-- r016_full_bounded_transport_warp_residual_unet_nc20_d6_height_light_geometric_lr0.0004_mas/
|   |   |   |-- r016_full_coord_basis_pressure_head_unet_nc20_d6_height_sdf_light_geometric_lr0.0004_maske/
|   |   |   |-- r016_full_frame_evolution_conditioner_unet_nc24_d6_height_sdf_normal_none_lr0.0004_masked/
|   |   |   |-- r016_full_local_enhanced_selective_scan_unet_nc16_d5_height_sdf_normal_light_geometric_lr0/
|   |   |   |-- r016_full_multi_scale_fourier_basis_head_unet_nc20_d6_height_sdf_normal_light_geometric_lr/
|   |   |   |-- r016_full_multi_scale_fourier_basis_head_unet_nc24_d6_height_sdf_normal_light_geometric_lr/
|   |   |   |-- r016_full_multiwavelet_multigrid_detail_unet_nc20_d6_height_sdf_light_geometric_lr0.0004_m/
|   |   |   |-- r016_full_planar_multigrid_vcycle_unet_nc24_d7_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r016_full_shared_expert_center_adapter_unet_nc20_d6_height_light_geometric_lr0.0004_masked/
|   |   |   `-- r016_full_sparse_tt_adapter_router_unet_nc20_d6_height_none_lr0.0003_masked_l1/
|   |   |-- r017/
|   |   |   |-- r017_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r017_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0/
|   |   |   |-- r017_full_boundary_assimilated_spectral_residual_nc20_d6_height_sdf_normal_light_geometric/
|   |   |   |-- r017_full_boundary_path_message_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l1_gradient/
|   |   |   |-- r017_full_kan_fno_hybrid_nc16_d5_height_light_geometric_lr0.0005_masked_l1/
|   |   |   |-- r017_full_localized_integral_differential_nc20_d6_height_sdf_light_geometric_lr0.0004_mask/
|   |   |   |-- r017_full_mamba2d_global_mixer_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_masked_l/
|   |   |   |-- r017_full_multi_scale_fourier_basis_head_nc24_d6_height_sdf_light_geometric_lr0.0004_maske/
|   |   |   |-- r017_full_perceiver_latent_bottleneck_nc16_d6_height_sdf_light_geometric_lr0.0003_masked_l/
|   |   |   |-- r017_full_planar_multigrid_vcycle_unet_nc24_d7_height_sdf_normal_light_geometric_lr0.00035/
|   |   |   |-- r017_full_senseiver_pseudo_sensor_head_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   `-- r017_full_swin_unetr_nc20_d6_height_sdf_light_geometric_lr0.0004_masked_huber/
|   |   |-- r018/
|   |   |   |-- r018_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_none_lr0.0004_masked_l1_gr/
|   |   |   |-- r018_full_anisotropic_boundary_hybrid_nc24_d6_height_sdf_normal_light_geometric_lr0.00035/
|   |   |   |-- r018_full_anisotropic_kernel_operator_nc16_d5_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r018_full_boundary_assimilated_spectral_residual_nc24_d6_height_sdf_normal_light_geometric/
|   |   |   |-- r018_full_fourier_lowrank_decoder_adapter_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r018_full_hadamard_gated_longconv_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r018_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r018_full_multi_scale_fourier_basis_head_nc20_d6_height_sdf_normal_light_geometric_lr0.000/
|   |   |   |-- r018_full_planar_multigrid_vcycle_nc28_d7_height_sdf_normal_light_geometric_lr0.0003_maske/
|   |   |   |-- r018_full_soft_boundary_token_mixer_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.000/
|   |   |   |-- r018_full_tt_adapter_decoder_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_maske/
|   |   |   `-- r018_full_uncertainty_point_boundary_refiner_unet_nc20_d6_height_sdf_normal_light_geometri/
|   |   |-- r019/
|   |   |   |-- r019_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_flip_rot_lr0.0004_masked_l/
|   |   |   |-- r019_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r019_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_none_lr0.00045_masked/
|   |   |   |-- r019_full_boundary_token_alltoall_mixer_unet_nc20_d6_height_sdf_normal_none_lr0.0004_maske/
|   |   |   |-- r019_full_coarse_spectral_residual_decoder_unet_nc20_d6_height_sdf_none_lr0.0004_masked_l1/
|   |   |   |-- r019_full_cross_shape_meta_adapter_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r019_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r019_full_hypernet_coord_lowrank_residual_unet_nc24_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r019_full_multi_scale_fourier_basis_head_unet_nc24_d6_height_sdf_none_lr0.0004_masked_l1_g/
|   |   |   |-- r019_full_planar_multigrid_vcycle_unet_nc24_d7_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r019_full_scale_adaptive_patch_consensus_unet_nc20_d6_height_sdf_normal_none_lr0.0004_mask/
|   |   |   `-- r019_full_terrain_conditioned_local_attention_unet_nc20_d6_height_sdf_normal_none_lr0.0004/
|   |   |-- r020/
|   |   |   |-- r020_full_afno_differential_detail_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r020_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r020_full_anisotropic_boundary_hybrid_nc20_d7_height_sdf_normal_flip_rot_lr0.00035_masked/
|   |   |   |-- r020_full_anisotropic_kernel_operator_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0/
|   |   |   |-- r020_full_boundary_assimilated_spectral_residual_unet_nc20_d6_height_sdf_normal_light_geom/
|   |   |   |-- r020_full_differential_consistency_pressure_head_unet_nc20_d6_height_sdf_normal_light_geom/
|   |   |   |-- r020_full_distilled_compact_inr_pressure_head_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |   |   |-- r020_full_dynamic_hybrid_operation_adapter_unet_nc20_d6_height_sdf_normal_light_geometric/
|   |   |   |-- r020_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r020_full_hypernet_coord_lowrank_residual_unet_nc20_d7_height_sdf_normal_light_geometric_l/
|   |   |   |-- r020_full_selective_mamba_encoder_conv_decoder_unet_nc16_d5_height_sdf_normal_light_geomet/
|   |   |   `-- r020_full_terrain_conditioned_local_attention_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |   |-- r021/
|   |   |   |-- r021_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_gradient_e/
|   |   |   |-- r021_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r021_full_anisotropic_boundary_hybrid_wtconv_nc20_d6_height_sdf_normal_light_geometric_lr0/
|   |   |   |-- r021_full_hypernet_coord_lowrank_residual_unet_geomtok_nc20_d6_height_sdf_normal_light_geo/
|   |   |   |-- r021_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r021_full_learned_height_warp_fourier_adapter_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |   |   |-- r021_full_multiwavelet_vcycle_detail_adapter_unet_nc20_d6_height_sdf_normal_light_geometri/
|   |   |   |-- r021_full_semantic_neighbor_key_ssm_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.000/
|   |   |   |-- r021_full_tensorized_fourier_local_global_adapter_unet_nc20_d6_height_sdf_normal_light_geo/
|   |   |   |-- r021_full_terrain_conditioned_local_attention_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |   |   `-- r021_full_unet_v3_lrbasis_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_masked_l1_gra/
|   |   |-- r022/
|   |   |   |-- r022_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0003_m/
|   |   |   |-- r022_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r022_full_anisotropic_boundary_hybrid_nc24_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |   |   |-- r022_full_attentive_state_space_restoration_unet_nc20_d6_height_sdf_normal_light_geometric/
|   |   |   |-- r022_full_boundary_graph_surrogate_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r022_full_hard_case_focal_residual_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r022_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r022_full_hypernet_coord_lowrank_residual_unet_nc24_d6_height_sdf_normal_light_geometric_l/
|   |   |   |-- r022_full_local_implicit_image_function_unet_nc20_d6_height_sdf_normal_light_geometric_lr0/
|   |   |   |-- r022_full_multiscale_residual_fpn_unet_nc20_d6_height_sdf_normal_light_geometric_lr0.0004/
|   |   |   |-- r022_full_terrain_conditioned_local_attention_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |   |   `-- r022_full_warped_anisotropic_boundary_hybrid_unet_nc20_d6_height_sdf_normal_light_geometri/
|   |   `-- r023/
|   |       |-- r023_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_heavy_geometric_lr0.0004_m/
|   |       |-- r023_full_anisotropic_boundary_hybrid_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_m/
|   |       |-- r023_full_beno_film_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_masked_l1_gradient/
|   |       |-- r023_full_boundary_saliency_gated_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_maske/
|   |       |-- r023_full_corrdiff_mean_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_masked_l1_gradi/
|   |       |-- r023_full_hypernet_coord_lowrank_residual_unet_nc20_d6_height_sdf_normal_light_geometric_l/
|   |       |-- r023_full_mambairv2_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_masked_l1_gradient/
|   |       |-- r023_full_segformer_allmlp_decoder_nc20_d6_height_sdf_normal_light_geometric_lr0.0004_mask/
|   |       |-- r023_full_terrain_conditioned_local_attention_unet_nc20_d6_height_sdf_normal_light_geometr/
|   |       `-- r023_full_terrain_conditioned_local_attention_unet_nc20_d7_height_sdf_normal_light_geometr/
|   |-- models/
|   |   |-- generated/
|   |   |   |-- adaptive_afno_unet.py
|   |   |   |-- adaptive_boundary_residual_render_unet.py
|   |   |   |-- afno_differential_detail_unet.py
|   |   |   |-- allaround_strip_mixer_unet.py
|   |   |   |-- anisotropic_boundary_hybrid.py
|   |   |   |-- anisotropic_boundary_hybrid_unet.py
|   |   |   |-- anisotropic_boundary_hybrid_wtconv.py
|   |   |   |-- anisotropic_kernel_operator.py
|   |   |   |-- anisotropic_kernel_operator_unet.py
|   |   |   |-- anisotropic_vcycle_fusion_unet.py
|   |   |   |-- asym_lora_unet.py
|   |   |   |-- attentive_state_space_restoration_unet.py
|   |   |   |-- attn_ssm_neighbor_unet.py
|   |   |   |-- attn_ssm_skip_unet.py
|   |   |   |-- band_adaptive_spectral_adapter_unet.py
|   |   |   |-- beno_film.py
|   |   |   |-- beno_multitoken_film_unet.py
|   |   |   |-- boundary_alltoall_surface_mixer_unet.py
|   |   |   |-- boundary_assimilated_spectral_residual.py
|   |   |   |-- boundary_assimilated_spectral_residual_unet.py
|   |   |   |-- boundary_coarse_graph_adapter_unet.py
|   |   |   |-- boundary_gated_multiscale_unet.py
|   |   |   |-- boundary_graph_surrogate_unet.py
|   |   |   |-- boundary_multipole_token_unet.py
|   |   |   |-- boundary_path_message.py
|   |   |   |-- boundary_path_message_unet.py
|   |   |   |-- boundary_pressure_basis_unet.py
|   |   |   |-- boundary_ring_mixer_unet.py
|   |   |   |-- boundary_saliency_gated.py
|   |   |   |-- boundary_strip_graph_mixer_unet.py
|   |   |   |-- boundary_token_alltoall_mixer_unet.py
|   |   |   |-- boundary_token_film_unet.py
|   |   |   |-- boundary_token_mixer_unet.py
|   |   |   |-- boundary_wtconv_large_receptive_unet.py
|   |   |   |-- bounded_transport_warp_residual_unet.py
|   |   |   |-- channel_transposed_attention_head_unet.py
|   |   |   |-- cnn_deeponet_lowrank.py
|   |   |   |-- coarse_grid_mp_unet.py
|   |   |   |-- coarse_interp_residual_head_unet.py
|   |   |   |-- coarse_prior_adain_unet.py
|   |   |   |-- coarse_spectral_residual_decoder_unet.py
|   |   |   |-- coarse_to_fine_ladder_unet.py
|   |   |   |-- colora_decoder_adapter_unet.py
|   |   |   |-- conditional_basis_decoder_mixer_unet.py
|   |   |   |-- coord_basis_pressure_head_unet.py
|   |   |   |-- coord_field_unet.py
|   |   |   |-- coordinate_moe_implicit_residual_head_unet.py
|   |   |   |-- corrdiff_mean.py
|   |   |   |-- cross_scale_operator_token_unet.py
|   |   |   |-- cross_shape_axial_adapter_unet.py
|   |   |   |-- cross_shape_decoder_adapter_unet.py
|   |   |   |-- cross_shape_meta_adapter_unet.py
|   |   |   |-- crossaxis_attn_unet.py
|   |   |   |-- dct_spectral_unet.py
|   |   |   |-- derived_feature_aggregate_unet.py
|   |   |   |-- differentiable_stencil_residual_unet.py
|   |   |   |-- differential_consistency_pressure_head_unet.py
|   |   |   |-- distilled_compact_inr_pressure_head_unet.py
|   |   |   |-- dual_branch_boundary_film_unet.py
|   |   |   |-- dual_spatial_channel_agg_unet.py
|   |   |   |-- dynamic_hybrid_operation_adapter_unet.py
|   |   |   |-- dynamic_roughness_dispatch_unet.py
|   |   |   |-- edge_offset_corrector_unet.py
|   |   |   |-- edsr_residual_head_unet.py
|   |   |   |-- encoder_mamba_residual_unet.py
|   |   |   |-- fg_moe_partition_unet.py
|   |   |   |-- fgmoe_decoder_unet.py
|   |   |   |-- fourier_lowrank_decoder_adapter_unet.py
|   |   |   |-- fourier_split_residual_head_unet.py
|   |   |   |-- fourier_unet.py
|   |   |   |-- frame_evolution_conditioner_unet.py
|   |   |   |-- frequency_adaptive_spectral_bottleneck_unet.py
|   |   |   |-- fusion_basis_decoder_unet.py
|   |   |   |-- geo_gate_conv_unet.py
|   |   |   |-- geometry_landmark_bottleneck_operator_unet.py
|   |   |   |-- geometry_landmark_operator_token_unet.py
|   |   |   |-- geometry_modulated_afno_unet.py
|   |   |   |-- gradient_boundary_token_alltoall_unet.py
|   |   |   |-- hadamard_gated_longconv_unet.py
|   |   |   |-- haloed_local_fourier_unet.py
|   |   |   |-- hard_case_focal_residual_unet.py
|   |   |   |-- height_derived_boundary_alltoall_mixer_unet.py
|   |   |   |-- height_only_anisotropic_compact_unet.py
|   |   |   |-- heterogeneous_micro_adapter_mixture_unet.py
|   |   |   |-- high_preservation_dual_aggregation_unet.py
|   |   |   |-- hilo_attn_unet.py
|   |   |   |-- hrnet.py
|   |   |   |-- hydra_lora_unet.py
|   |   |   |-- hyperinr_residual_head_unet.py
|   |   |   |-- hypernet_adapter_unet.py
|   |   |   |-- hypernet_coord_lowrank_residual_unet.py
|   |   |   |-- hypernet_coord_lowrank_residual_unet_geomtok.py
|   |   |   |-- incremental_mode_gate_afno_unet.py
|   |   |   |-- internal_backprojection_ladder_unet.py
|   |   |   |-- iterative_residual_polisher_unet.py
|   |   |   |-- kan_fno_hybrid.py
|   |   |   |-- laplace_rational_filter_unet.py
|   |   |   |-- latent_cross_attention_operator_unet.py
|   |   |   |-- latent_graph_processor_unet.py
|   |   |   |-- latent_grid_adapter_unet.py
|   |   |   |-- learned_height_warp_fourier_adapter_unet.py
|   |   |   |-- liif_head_unet.py
|   |   |   |-- local_enhanced_selective_scan_unet.py
|   |   |   |-- local_implicit_image_function_unet.py
|   |   |   |-- localized_integral_differential.py
|   |   |   |-- localized_integral_differential_unet.py
|   |   |   |-- lowrank_spatial_operator_unet.py
|   |   |   |-- mamba2d_global_mixer.py
|   |   |   |-- mambairv2.py
|   |   |   |-- mean_residual_corrector_unet.py
|   |   |   |-- mlp_multiscale_decoder_unet.py
|   |   |   |-- msca_unet.py
|   |   |   |-- multi_scale_fourier_basis_head.py
|   |   |   |-- multi_scale_fourier_basis_head_unet.py
|   |   |   |-- multigrid_spectral_unet.py
|   |   |   |-- multiscale_linear_attention_decoder_unet.py
|   |   |   |-- multiscale_linear_attention_skip_mixer_unet.py
|   |   |   |-- multiscale_residual_fpn_unet.py
|   |   |   |-- multiscale_ssm_pyramid_unet.py
|   |   |   |-- multiwavelet_multigrid_detail_unet.py
|   |   |   |-- multiwavelet_vcycle_detail_adapter_unet.py
|   |   |   |-- nafnet.py
|   |   |   |-- noncausal_neighbor_ssm_unet.py
|   |   |   |-- one_sided_lora_output_adapter_unet.py
|   |   |   |-- overlap_add_spectral_adapter_unet.py
|   |   |   |-- patch_consensus_residual_unet.py
|   |   |   |-- patchwise_sparse_conv_moe_unet.py
|   |   |   |-- perceiver_latent_bottleneck.py
|   |   |   |-- physics_state_slice_mixer_unet.py
|   |   |   |-- pid_dcb_unet.py
|   |   |   |-- planar_multigrid_vcycle.py
|   |   |   |-- planar_multigrid_vcycle_unet.py
|   |   |   |-- pseudo_station_bottleneck_fusion_unet.py
|   |   |   |-- reduced_kv_context_unet.py
|   |   |   |-- residual_expert_codebook_mixer_unet.py
|   |   |   |-- restormer_unet.py
|   |   |   |-- roughness_gated_ssm_unet.py
|   |   |   |-- scale_adaptive_patch_consensus_unet.py
|   |   |   |-- sdf_gradient_field_conditioned_film_unet.py
|   |   |   |-- segformer_allmlp_decoder.py
|   |   |   |-- selective_mamba_encoder_conv_decoder_unet.py
|   |   |   |-- semantic_neighbor_key_ssm_unet.py
|   |   |   |-- semi_lagrangian_warp_unet.py
|   |   |   |-- senseiver_pseudo_sensor_head.py
|   |   |   |-- separable_fourier_unet.py
|   |   |   |-- shared_expert_center_adapter_unet.py
|   |   |   |-- shared_norm_cross_shape_adapter_unet.py
|   |   |   |-- soft_boundary_token_mixer_unet.py
|   |   |   |-- sparse_residual_refinement_unet.py
|   |   |   |-- sparse_shared_projection_expert_adapter_unet.py
|   |   |   |-- sparse_tt_adapter_router_unet.py
|   |   |   |-- spatial_prior_adapter_unet.py
|   |   |   |-- squeeze_axial_detail_unet.py
|   |   |   |-- swin_unetr.py
|   |   |   |-- tensorized_fourier_local_global_adapter_unet.py
|   |   |   |-- terrain_conditioned_local_attention_unet.py
|   |   |   |-- texture_boundary_dual_fusion_unet.py
|   |   |   |-- texture_energy_modulated_ssm_unet.py
|   |   |   |-- tiled_spectral_mixer_unet.py
|   |   |   |-- transolver_lite.py
|   |   |   |-- tt_adapter_decoder_unet.py
|   |   |   |-- uncertainty_point_boundary_refiner_unet.py
|   |   |   |-- unet_sdf_7level.py
|   |   |   |-- unet_v3_gausres.py
|   |   |   |-- unet_v3_geomtok.py
|   |   |   |-- unet_v3_lrbasis.py
|   |   |   |-- unet_v3_pointrefine.py
|   |   |   |-- visual_context_replay_unet.py
|   |   |   |-- vmt_lite_shared_projection_unet.py
|   |   |   |-- vq_pressure_regime_codebook_unet.py
|   |   |   |-- warped_anisotropic_boundary_hybrid_unet.py
|   |   |   |-- wavelet_gated_anisotropic.py
|   |   |   `-- wavelet_residual_decoder_unet.py
|   |   |-- helpers/
|   |   |   |-- __init__.py
|   |   |   |-- afno_block.py
|   |   |   |-- base.py
|   |   |   |-- convnext_v2_unet.py
|   |   |   |-- ffno.py
|   |   |   |-- fno_v3.py
|   |   |   `-- ufno.py
|   |   |-- tier_a_unet_variants/
|   |   |   |-- attention_gate_unet.py
|   |   |   |-- cbam_unet.py
|   |   |   |-- dcn_unet.py
|   |   |   |-- dilated_unet.py
|   |   |   |-- hrnet.py
|   |   |   |-- kan_unet.py
|   |   |   |-- nafnet.py
|   |   |   `-- sac_unet.py
|   |   |-- tier_b_new_architectures/
|   |   |   |-- cnn_deeponet.py
|   |   |   |-- hrformer.py
|   |   |   |-- mamba2d.py
|   |   |   |-- perceiver.py
|   |   |   |-- quadmamba.py
|   |   |   |-- swin_unetr.py
|   |   |   |-- transolver.py
|   |   |   `-- umamba.py
|   |   |-- tier_c_operators/
|   |   |   |-- afno.py
|   |   |   |-- cno.py
|   |   |   |-- fno2d.py
|   |   |   |-- transolver_lite.py
|   |   |   `-- uno.py
|   |   |-- tier_d_hybrids/
|   |   |   |-- attention_mamba.py
|   |   |   |-- dilated_fno.py
|   |   |   |-- dilated_hrformer.py
|   |   |   |-- fno_encoder_decoder.py
|   |   |   |-- fourier_unet.py
|   |   |   |-- hrdcn.py
|   |   |   |-- mamba_attention.py
|   |   |   |-- multiscale_conv.py
|   |   |   |-- residual_spectral.py
|   |   |   `-- sac_mamba.py
|   |   `-- __init__.py
|   |-- scripts/
|   |-- shared/
|   |   |-- configs/
|   |   |   |-- __init__.py
|   |   |   |-- problem_definition.yaml
|   |   |   |-- schema.py
|   |   |   `-- search_space.json
|   |   |-- models/
|   |   |   |-- __init__.py
|   |   |   |-- afno.py
|   |   |   |-- afno_block.py
|   |   |   |-- attention_gate_unet.py
|   |   |   |-- attention_mamba.py
|   |   |   |-- base.py
|   |   |   |-- cbam_unet.py
|   |   |   |-- cnn_deeponet.py
|   |   |   |-- cno.py
|   |   |   |-- convnext_v2_unet.py
|   |   |   |-- dcn_unet.py
|   |   |   |-- dilated_fno.py
|   |   |   |-- dilated_hrformer.py
|   |   |   |-- dilated_unet.py
|   |   |   |-- ffno.py
|   |   |   |-- fno2d.py
|   |   |   |-- fno_encoder_decoder.py
|   |   |   |-- fno_v3.py
|   |   |   |-- fourier_unet.py
|   |   |   |-- hrdcn.py
|   |   |   |-- hrformer.py
|   |   |   |-- hrnet.py
|   |   |   |-- kan_unet.py
|   |   |   |-- mamba2d.py
|   |   |   |-- mamba_attention.py
|   |   |   |-- multiscale_conv.py
|   |   |   |-- nafnet.py
|   |   |   |-- perceiver.py
|   |   |   |-- quadmamba.py
|   |   |   |-- residual_spectral.py
|   |   |   |-- sac_mamba.py
|   |   |   |-- sac_unet.py
|   |   |   |-- swin_unetr.py
|   |   |   |-- transolver.py
|   |   |   |-- transolver_lite.py
|   |   |   |-- ufno.py
|   |   |   |-- umamba.py
|   |   |   |-- unet_afno.py
|   |   |   |-- unet_sdf_7level.py
|   |   |   |-- unet_v3.py
|   |   |   `-- uno.py
|   |   |-- eval_module.py
|   |   |-- losses.py
|   |   `-- train.py
|   |-- model_rounds_manifest.csv
|   `-- README.md
|-- hybrid_workflow_full/
|   |-- configs/
|   |   |-- candidate_library.json
|   |   |-- explorer_config.json
|   |   |-- model_specs.yaml
|   |   |-- search_space.json
|   |   `-- suggestion_round1.json
|   |-- engine/
|   |   |-- __init__.py
|   |   |-- analyzer.py
|   |   |-- executor.py
|   |   |-- runner.py
|   |   `-- state_manager.py
|   |-- explorer/
|   |   |-- __init__.py
|   |   |-- ai_callers.py
|   |   |-- candidate_library.py
|   |   |-- codegen.py
|   |   |-- explorer.py
|   |   |-- modes.py
|   |   |-- planner.py
|   |   |-- planner_v4.py
|   |   |-- prompt_builder.py
|   |   |-- proposal_collector.py
|   |   |-- reviewer.py
|   |   `-- suggester.py
|   |-- initial_knowledge/
|   |   |-- INITIAL_KNOWLEDGE.md
|   |   `-- model_results.csv
|   |-- scripts/
|   |-- tests/
|   |   `-- test_r13_planner_synthesis_gate.py
|   |-- workflow_engine/
|   |   |-- tests/
|   |   |   |-- fixtures/
|   |   |   `-- test_codegen_hardening.py
|   |   |-- __init__.py
|   |   |-- attempt_manifest.py
|   |   |-- failure_classifier.py
|   |   |-- repair_phase9_manifest.py
|   |   |-- schema_guards.py
|   |   |-- test_attempt_budget_fix.py
|   |   |-- test_experiment_id_uniqueness.py
|   |   |-- test_proposal_accountability.py
|   |   |-- test_resource_guard.py
|   |   |-- test_schema_guard_loss_fallback.py
|   |   |-- validate_web_scout_prompts.py
|   |   |-- workflow_codegen.py
|   |   |-- workflow_common.py
|   |   |-- workflow_controller.py
|   |   |-- workflow_executor.py
|   |   |-- workflow_knowledge.py
|   |   |-- workflow_planner.py
|   |   |-- workflow_reviewer.py
|   |   `-- workflow_runner.py
|   |-- INITIAL_KNOWLEDGE_PACKAGE_TEMPLATE.md
|   |-- LOCKED_FILES.md
|   |-- README.md
|   `-- run_hybrid_workflow_runner.bat
|-- prompts_and_prompt_generators/
|   |-- prompt_templates_and_builders/
|   |   |-- explorer/
|   |   |   |-- ai_callers.py
|   |   |   |-- codegen.py
|   |   |   |-- modes.py
|   |   |   |-- planner.py
|   |   |   |-- planner_v4.py
|   |   |   |-- prompt_builder.py
|   |   |   |-- proposal_collector.py
|   |   |   `-- reviewer.py
|   |   |-- scripts/
|   |   |   `-- run_campaign.py
|   |   `-- workflow_engine/
|   |       |-- tests/
|   |       |-- validate_web_scout_prompts.py
|   |       |-- workflow_codegen.py
|   |       |-- workflow_planner.py
|   |       `-- workflow_reviewer.py
|   |-- representative_full_prompts/
|   |   |-- r000/
|   |   |   |-- external_ideas_codex_web_prompt.txt
|   |   |   |-- external_ideas_review_claude_prompt.txt
|   |   |   |-- planner_prompt.txt
|   |   |   |-- planner_prompt_codex.txt
|   |   |   |-- planner_prompt_gemini.txt
|   |   |   |-- quality_gate_codex_prompt.txt
|   |   |   `-- synthesis_claude_prompt.txt
|   |   |-- r010/
|   |   |   |-- external_ideas_codex_web_prompt.txt
|   |   |   |-- external_ideas_review_claude_prompt.txt
|   |   |   |-- planner_prompt.txt
|   |   |   |-- planner_prompt_codex.txt
|   |   |   |-- planner_prompt_gemini.txt
|   |   |   |-- quality_gate_codex_prompt.txt
|   |   |   `-- synthesis_claude_prompt.txt
|   |   `-- r024/
|   |       |-- external_ideas_codex_web_prompt.txt
|   |       |-- external_ideas_review_claude_prompt.txt
|   |       |-- planner_prompt.txt
|   |       |-- planner_prompt_claude.txt
|   |       |-- planner_prompt_codex.txt
|   |       |-- planner_prompt_gemini.txt
|   |       |-- quality_gate_codex_prompt.txt
|   |       `-- synthesis_claude_prompt.txt
|   |-- prompt_manifest.csv
|   `-- README.md
|-- results_summary/
|   |-- grid_summary/
|   |   |-- grid_all_metrics_summary.csv
|   |   |-- grid_top50_metrics_summary.csv
|   |   |-- holdout_results.json
|   |   `-- PACKAGE_MANIFEST.json
|   |-- hybrid_corrected_val55/
|   |   |-- aggregate_corrected_val55_best.csv
|   |   |-- per_seed_corrected_val55_best.csv
|   |   `-- SUMMARY.md
|   `-- sequential_summary/
|       |-- phase9_final_report_v3.tex
|       `-- phase9_lu_metrics_summary.json
|-- sequential_training_models/
|   |-- model_rounds/
|   |   |-- r002/
|   |   |   |-- configs/
|   |   |   |-- model_00_cbam_unet/
|   |   |   |-- model_01_fourier_unet/
|   |   |   |-- model_02_hrdcn/
|   |   |   |-- model_03_hrformer/
|   |   |   |-- model_04_mamba_attention/
|   |   |   |-- model_05_multiscale_conv/
|   |   |   |-- model_06_residual_spectral/
|   |   |   |-- model_07_swin_unetr/
|   |   |   |-- model_08_transolver_lite/
|   |   |   |-- model_09_transolver/
|   |   |   `-- model_10_ufno/
|   |   |-- r003/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cbam_unet/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dcn_unet/
|   |   |   |-- model_04_dilated_unet/
|   |   |   |-- model_05_residual_spectral/
|   |   |   |-- model_06_residual_spectral/
|   |   |   `-- model_07_sac_unet/
|   |   |-- r004/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_dilated_unet/
|   |   |   |-- model_02_fno_encoder_decoder/
|   |   |   |-- model_03_hrnet/
|   |   |   |-- model_04_residual_spectral/
|   |   |   |-- model_05_residual_spectral/
|   |   |   |-- model_06_residual_spectral/
|   |   |   |-- model_07_sac_unet/
|   |   |   |-- model_08_sac_unet/
|   |   |   `-- model_09_ufno/
|   |   |-- r005/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cbam_unet/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_hrdcn/
|   |   |   |-- model_05_multiscale_conv/
|   |   |   |-- model_06_quadmamba/
|   |   |   |-- model_07_residual_spectral/
|   |   |   |-- model_08_sac_unet/
|   |   |   `-- model_09_sac_unet/
|   |   |-- r006/
|   |   |   |-- configs/
|   |   |   |-- model_00_dilated_fno/
|   |   |   |-- model_01_mamba2d/
|   |   |   |-- model_02_nafnet/
|   |   |   |-- model_03_perceiver_io/
|   |   |   |-- model_04_quadmamba/
|   |   |   |-- model_05_residual_spectral/
|   |   |   |-- model_06_residual_spectral/
|   |   |   |-- model_07_sac_unet/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_09_umamba/
|   |   |   `-- model_10_unet_afno/
|   |   |-- r007/
|   |   |   |-- configs/
|   |   |   |-- model_00_convnext_v2_unet/
|   |   |   |-- model_01_dilated_unet/
|   |   |   |-- model_02_mamba2d/
|   |   |   |-- model_03_quadmamba/
|   |   |   |-- model_04_residual_spectral/
|   |   |   |-- model_05_sac_mamba/
|   |   |   |-- model_06_sac_unet/
|   |   |   |-- model_07_swin_unetr/
|   |   |   `-- model_11_unet_v3/
|   |   |-- r008/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_fourier_unet/
|   |   |   |-- model_02_mamba2d/
|   |   |   |-- model_03_quadmamba/
|   |   |   |-- model_04_residual_spectral/
|   |   |   |-- model_05_sac_unet/
|   |   |   `-- model_10_unet_v3/
|   |   |-- r009/
|   |   |   |-- configs/
|   |   |   |-- model_00_cnn_deeponet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_dilated_hrformer/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_fno_v3/
|   |   |   |-- model_05_hrnet/
|   |   |   |-- model_06_mamba2d/
|   |   |   |-- model_07_sac_unet/
|   |   |   |-- model_08_umamba/
|   |   |   `-- model_10_unet_v3/
|   |   |-- r010/
|   |   |   |-- configs/
|   |   |   |-- model_00_dilated_unet/
|   |   |   |-- model_01_hrformer/
|   |   |   |-- model_02_mamba2d/
|   |   |   |-- model_03_mamba2d/
|   |   |   |-- model_04_quadmamba/
|   |   |   |-- model_05_residual_spectral/
|   |   |   |-- model_06_sac_unet/
|   |   |   |-- model_07_swin_unet/
|   |   |   `-- model_10_unet_v3/
|   |   |-- r011/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno_block/
|   |   |   |-- model_01_attention_mamba/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_fno/
|   |   |   |-- model_04_fno_encoder_decoder/
|   |   |   |-- model_05_kan_unet/
|   |   |   |-- model_06_mamba2d/
|   |   |   |-- model_07_quadmamba/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_09_unet_afno/
|   |   |   `-- model_11_unet_v3/
|   |   |-- r012/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno_block/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_dilated_unet/
|   |   |   |-- model_03_mamba2d/
|   |   |   |-- model_04_mamba2d/
|   |   |   |-- model_05_quadmamba/
|   |   |   |-- model_06_sac_unet/
|   |   |   |-- model_07_sac_unet/
|   |   |   `-- model_10_unet_v3/
|   |   |-- r013/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno_block/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_mamba2d/
|   |   |   |-- model_05_mamba2d/
|   |   |   |-- model_06_quadmamba/
|   |   |   |-- model_07_residual_spectral/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_09_sac_unet/
|   |   |   `-- model_11_unet_v3/
|   |   |-- r014/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno_block/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_mamba2d/
|   |   |   |-- model_03_multiscale_conv/
|   |   |   |-- model_04_perceiver_io/
|   |   |   |-- model_05_quadmamba/
|   |   |   |-- model_06_residual_spectral/
|   |   |   |-- model_07_sac_unet/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_10_unet_v3/
|   |   |   `-- model_11_uno/
|   |   |-- r015/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cbam_unet/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_mamba2d/
|   |   |   |-- model_06_nafnet/
|   |   |   |-- model_07_quadmamba/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_09_transolver_lite/
|   |   |   `-- model_10_unet_afno/
|   |   |-- r016/
|   |   |   |-- configs/
|   |   |   |-- model_00_cno/
|   |   |   |-- model_01_dilated_hrformer/
|   |   |   |-- model_02_ffno/
|   |   |   |-- model_03_hrdcn/
|   |   |   |-- model_04_hrnet/
|   |   |   |-- model_05_mamba2d/
|   |   |   |-- model_06_mamba_attention/
|   |   |   |-- model_07_quadmamba/
|   |   |   |-- model_08_sac_unet/
|   |   |   |-- model_09_sac_unet/
|   |   |   `-- model_10_uno/
|   |   |-- r017/
|   |   |   |-- configs/
|   |   |   |-- model_00_cno/
|   |   |   |-- model_01_fourier_unet/
|   |   |   |-- model_02_hrdcn/
|   |   |   |-- model_03_mamba2d/
|   |   |   |-- model_04_mamba2d/
|   |   |   |-- model_05_quadmamba/
|   |   |   |-- model_06_sac_unet/
|   |   |   `-- model_07_umamba/
|   |   |-- r018/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_mamba2d/
|   |   |   |-- model_06_quadmamba/
|   |   |   `-- model_07_sac_unet/
|   |   |-- r019/
|   |   |   |-- configs/
|   |   |   |-- model_00_dcn_unet/
|   |   |   |-- model_01_dilated_unet/
|   |   |   |-- model_02_fourier_unet/
|   |   |   |-- model_03_mamba2d/
|   |   |   |-- model_04_multiscale_conv/
|   |   |   |-- model_05_sac_unet/
|   |   |   |-- model_06_sac_unet/
|   |   |   `-- model_07_transolver_lite/
|   |   |-- r020/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_dilated_unet/
|   |   |   |-- model_03_ffno/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_hrnet/
|   |   |   |-- model_06_nafnet/
|   |   |   `-- model_07_sac_unet/
|   |   |-- r021/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dilated_unet/
|   |   |   |-- model_04_fourier_unet/
|   |   |   |-- model_05_quadmamba/
|   |   |   `-- model_06_sac_unet/
|   |   |-- r022/
|   |   |   |-- configs/
|   |   |   |-- model_00_afno_block/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_dcn_unet/
|   |   |   |-- model_04_mamba2d/
|   |   |   |-- model_05_sac_unet/
|   |   |   |-- model_06_sac_unet/
|   |   |   |-- model_07_transolver_lite/
|   |   |   `-- model_08_unet_afno/
|   |   |-- r023/
|   |   |   |-- configs/
|   |   |   |-- model_00_attention_gate_unet/
|   |   |   |-- model_01_attention_gate_unet/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_cno/
|   |   |   |-- model_04_dilated_unet/
|   |   |   `-- model_05_fno_encoder_decoder/
|   |   |-- r024/
|   |   |   |-- configs/
|   |   |   |-- model_00_boundary_crossattn_unet/
|   |   |   |-- model_01_boundary_film_unet/
|   |   |   |-- model_02_cno/
|   |   |   |-- model_03_geometry_bridge_unet/
|   |   |   |-- model_04_lowrank_kernel_unet/
|   |   |   |-- model_05_mamba2d/
|   |   |   |-- model_06_sac_unet/
|   |   |   |-- model_07_triscale_adapter_unet/
|   |   |   `-- model_10_wavelet_refine_unet/
|   |   |-- r025/
|   |   |   |-- configs/
|   |   |   |-- model_00_boundary_alignment_adapter_unet/
|   |   |   |-- model_01_boundary_crossattn_unet/
|   |   |   |-- model_02_boundary_crossattn_unet/
|   |   |   |-- model_03_boundary_film_unet/
|   |   |   |-- model_04_compressed_geometry_latent_unet/
|   |   |   |-- model_05_coord_residual_unet/
|   |   |   |-- model_06_expansion_separable_unet/
|   |   |   |-- model_07_gaussian_residual_unet/
|   |   |   |-- model_08_geometry_bridge_unet/
|   |   |   |-- model_09_self_gated_boundary_refine_unet/
|   |   |   `-- model_10_wavelet_refine_unet/
|   |   |-- r026/
|   |   |   |-- configs/
|   |   |   |-- model_00_boundary_crossattn_unet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_corrdiff_residual_unet/
|   |   |   |-- model_03_feature_warp_residual_unet/
|   |   |   |-- model_04_finegrained_moe_adapter_unet/
|   |   |   |-- model_05_geometry_bridge_unet/
|   |   |   |-- model_06_liif_residual_decoder_unet/
|   |   |   |-- model_07_lora_residual_adapter_unet/
|   |   |   |-- model_08_self_gated_boundary_refine_unet/
|   |   |   `-- model_09_wavelet_refine_unet/
|   |   |-- r027/
|   |   |   |-- configs/
|   |   |   |-- model_00_adaptive_frequency_modulation_unet/
|   |   |   |-- model_01_boundary_crossattn_unet/
|   |   |   |-- model_02_boundary_crossattn_unet/
|   |   |   |-- model_03_channel_sliced_moe_unet/
|   |   |   |-- model_04_geometry_bridge_unet/
|   |   |   |-- model_05_latent_grid_bridge_unet/
|   |   |   |-- model_06_mean_residual_decoder_unet/
|   |   |   |-- model_07_periodic_coord_residual_unet/
|   |   |   |-- model_08_self_gated_boundary_refine_unet/
|   |   |   |-- model_09_tensorized_spectral_adapter_unet/
|   |   |   |-- model_10_wavelet_refine_unet/
|   |   |   `-- model_11_wavelet_refine_unet/
|   |   |-- r028/
|   |   |   |-- configs/
|   |   |   |-- model_00_boundary_crossattn_unet/
|   |   |   |-- model_01_cno/
|   |   |   |-- model_02_coarse_to_fine_spectral_fusion_unet/
|   |   |   |-- model_03_evolution_detail_decomp_unet/
|   |   |   |-- model_04_geometry_bridge_unet/
|   |   |   |-- model_05_lora_residual_adapter_unet/
|   |   |   |-- model_06_mlp_cross_scale_decoder_unet/
|   |   |   |-- model_07_pid_boundary_detail_context_unet/
|   |   |   |-- model_08_strip_mixer_refine_unet/
|   |   |   |-- model_09_tensorized_spectral_adapter_unet/
|   |   |   `-- model_10_wavelet_refine_unet/
|   |   |-- r029/
|   |   |   |-- configs/
|   |   |   |-- model_00_axis_factor_spectral_adapter_unet/
|   |   |   |-- model_01_coarse_conditioned_detail_decoder_unet/
|   |   |   |-- model_02_coarse_to_fine_spectral_fusion_unet/
|   |   |   |-- model_03_content_aware_reassembly_unet/
|   |   |   |-- model_04_height_guided_pac_unet/
|   |   |   |-- model_05_hypercoord_residual_decoder_unet/
|   |   |   |-- model_06_physics_slice_adapter_unet/
|   |   |   |-- model_07_pid_boundary_detail_context_unet/
|   |   |   |-- model_08_tensorized_spectral_adapter_unet/
|   |   |   |-- model_09_tensorized_spectral_adapter_unet/
|   |   |   `-- model_11_wavelet_refine_unet/
|   |   |-- r030/
|   |   |   |-- configs/
|   |   |   |-- model_00_boundary_crossattn_unet/
|   |   |   |-- model_01_dual_aggregate_detail_gate_unet/
|   |   |   |-- model_02_hypercoord_microdecoder_unet/
|   |   |   |-- model_03_multiscale_memory_token_unet/
|   |   |   |-- model_04_shape_basis_residual_head_unet/
|   |   |   |-- model_05_shared_expert_center_router_unet/
|   |   |   |-- model_06_sparse_fourier_delta_adapter_unet/
|   |   |   |-- model_07_tensorized_spectral_adapter_unet/
|   |   |   |-- model_08_tensorized_spectral_adapter_unet/
|   |   |   |-- model_10_wavelet_refine_unet/
|   |   |   `-- model_11_wavelet_refine_unet/
|   |   `-- r031/
|   |       |-- configs/
|   |       |-- model_00_boundary_token_film_unet/
|   |       |-- model_01_dual_aggregate_detail_gate_unet/
|   |       |-- model_02_dual_aggregate_detail_gate_unet/
|   |       |-- model_03_hypercoord_microdecoder_unet/
|   |       |-- model_04_invariant_height_descriptor_unet/
|   |       |-- model_05_local_spectral_wavelet_operator_unet/
|   |       |-- model_06_lowrank_context_adapter_unet/
|   |       |-- model_07_omniscan_skip_mixer_unet/
|   |       |-- model_08_tensorized_spectral_adapter_unet/
|   |       |-- model_09_tensorized_spectral_adapter_unet/
|   |       `-- model_10_wavelet_refine_unet/
|   |-- models/
|   |   |-- generated/
|   |   |   |-- adaptive_frequency_modulation_unet.py
|   |   |   |-- adaptive_local_implicit_expert_unet.py
|   |   |   |-- afno_block.py
|   |   |   |-- axis_factor_spectral_adapter_unet.py
|   |   |   |-- base.py
|   |   |   |-- boundary_alignment_adapter_unet.py
|   |   |   |-- boundary_crossattn_unet.py
|   |   |   |-- boundary_film_unet.py
|   |   |   |-- boundary_token_film_unet.py
|   |   |   |-- channel_sliced_moe_unet.py
|   |   |   |-- cnn_deeponet.py
|   |   |   |-- cno.py
|   |   |   |-- cno_dilated_hybrid_v1.py
|   |   |   |-- coarse_conditioned_detail_decoder_unet.py
|   |   |   |-- coarse_to_fine_spectral_fusion_unet.py
|   |   |   |-- compressed_geometry_latent_unet.py
|   |   |   |-- content_aware_reassembly_unet.py
|   |   |   |-- convnext_v2_unet.py
|   |   |   |-- coord_residual_unet.py
|   |   |   |-- corrdiff_residual_unet.py
|   |   |   |-- deterministic_evolution_context_unet.py
|   |   |   |-- dilated_modulation_adapter_unet.py
|   |   |   |-- dual_aggregate_detail_gate_unet.py
|   |   |   |-- dynamic_kernel_bank_unet.py
|   |   |   |-- evolution_detail_decomp_unet.py
|   |   |   |-- expansion_separable_unet.py
|   |   |   |-- feature_warp_residual_unet.py
|   |   |   |-- ffno.py
|   |   |   |-- finegrained_moe_adapter_unet.py
|   |   |   |-- fno_v3.py
|   |   |   |-- gaussian_residual_unet.py
|   |   |   |-- geometry_bridge_unet.py
|   |   |   |-- global_descriptor_film_unet.py
|   |   |   |-- height_geomlift_unet.py
|   |   |   |-- height_guided_pac_unet.py
|   |   |   |-- hrformer.py
|   |   |   |-- hypercoord_microdecoder_unet.py
|   |   |   |-- hypercoord_residual_decoder_unet.py
|   |   |   |-- invariant_height_descriptor_unet.py
|   |   |   |-- latent_grid_bridge_unet.py
|   |   |   |-- liif_residual_decoder_unet.py
|   |   |   |-- local_spectral_wavelet_operator_unet.py
|   |   |   |-- lora_residual_adapter_unet.py
|   |   |   |-- lowrank_context_adapter_unet.py
|   |   |   |-- lowrank_kernel_unet.py
|   |   |   |-- mean_plus_residual_head_unet.py
|   |   |   |-- mean_residual_decoder_unet.py
|   |   |   |-- mlp_cross_scale_decoder_unet.py
|   |   |   |-- multiscale_memory_token_unet.py
|   |   |   |-- omniscan_skip_mixer_unet.py
|   |   |   |-- p5_res_unet.py
|   |   |   |-- p6_broken_unet.py
|   |   |   |-- p8_dcn_unet.py
|   |   |   |-- p8_dilated_unet.py
|   |   |   |-- p8_fourier_unet.py
|   |   |   |-- p8_kan_unet.py
|   |   |   |-- p8_sac_unet.py
|   |   |   |-- p8_unet_v3.py
|   |   |   |-- patch_tensorized_spectral_unet.py
|   |   |   |-- perceiver_io.py
|   |   |   |-- periodic_coord_residual_unet.py
|   |   |   |-- physics_slice_adapter_unet.py
|   |   |   |-- pid_boundary_detail_context_unet.py
|   |   |   |-- residual_spectral.py
|   |   |   |-- restormer_refine_unet.py
|   |   |   |-- sac_unet.py
|   |   |   |-- self_gated_boundary_refine_unet.py
|   |   |   |-- shape_basis_residual_head_unet.py
|   |   |   |-- shared_expert_center_router_unet.py
|   |   |   |-- sparse_fourier_delta_adapter_unet.py
|   |   |   |-- strip_mixer_refine_unet.py
|   |   |   |-- swin_unet.py
|   |   |   |-- swin_unet_lite.py
|   |   |   |-- tensorized_spectral_adapter_unet.py
|   |   |   |-- transolver.py
|   |   |   |-- transolver_lite.py
|   |   |   |-- triscale_adapter_unet.py
|   |   |   |-- umamba.py
|   |   |   |-- unet_afno.py
|   |   |   |-- unet_sdf_7level.py
|   |   |   |-- unet_v3.py
|   |   |   |-- uno.py
|   |   |   `-- wavelet_refine_unet.py
|   |   |-- helpers/
|   |   |   |-- __init__.py
|   |   |   |-- afno_block.py
|   |   |   |-- base.py
|   |   |   |-- convnext_v2_unet.py
|   |   |   |-- ffno.py
|   |   |   |-- fno_v3.py
|   |   |   `-- ufno.py
|   |   |-- tier_a_unet_variants/
|   |   |   |-- attention_gate_unet.py
|   |   |   |-- cbam_unet.py
|   |   |   |-- dcn_unet.py
|   |   |   |-- dilated_unet.py
|   |   |   |-- hrnet.py
|   |   |   |-- kan_unet.py
|   |   |   |-- nafnet.py
|   |   |   `-- sac_unet.py
|   |   |-- tier_b_new_architectures/
|   |   |   |-- cnn_deeponet.py
|   |   |   |-- hrformer.py
|   |   |   |-- mamba2d.py
|   |   |   |-- perceiver.py
|   |   |   |-- quadmamba.py
|   |   |   |-- swin_unetr.py
|   |   |   |-- transolver.py
|   |   |   `-- umamba.py
|   |   |-- tier_c_operators/
|   |   |   |-- afno.py
|   |   |   |-- cno.py
|   |   |   |-- fno2d.py
|   |   |   |-- transolver_lite.py
|   |   |   `-- uno.py
|   |   |-- tier_d_hybrids/
|   |   |   |-- attention_mamba.py
|   |   |   |-- dilated_fno.py
|   |   |   |-- dilated_hrformer.py
|   |   |   |-- fno_encoder_decoder.py
|   |   |   |-- fourier_unet.py
|   |   |   |-- hrdcn.py
|   |   |   |-- mamba_attention.py
|   |   |   |-- multiscale_conv.py
|   |   |   |-- residual_spectral.py
|   |   |   `-- sac_mamba.py
|   |   `-- __init__.py
|   |-- selected_round_configs/
|   |   |-- r002/
|   |   |   `-- full_configs/
|   |   |-- r003/
|   |   |   `-- full_configs/
|   |   |-- r004/
|   |   |   `-- full_configs/
|   |   |-- r005/
|   |   |   `-- full_configs/
|   |   |-- r006/
|   |   |   `-- full_configs/
|   |   |-- r007/
|   |   |   `-- full_configs/
|   |   |-- r008/
|   |   |   `-- full_configs/
|   |   |-- r009/
|   |   |   `-- full_configs/
|   |   |-- r010/
|   |   |   `-- full_configs/
|   |   |-- r011/
|   |   |   `-- full_configs/
|   |   |-- r012/
|   |   |   `-- full_configs/
|   |   |-- r013/
|   |   |   `-- full_configs/
|   |   |-- r014/
|   |   |   `-- full_configs/
|   |   |-- r015/
|   |   |   `-- full_configs/
|   |   |-- r016/
|   |   |   `-- full_configs/
|   |   |-- r017/
|   |   |   `-- full_configs/
|   |   |-- r018/
|   |   |   `-- full_configs/
|   |   |-- r019/
|   |   |   `-- full_configs/
|   |   |-- r020/
|   |   |   `-- full_configs/
|   |   |-- r021/
|   |   |   `-- full_configs/
|   |   |-- r022/
|   |   |   `-- full_configs/
|   |   |-- r023/
|   |   |   `-- full_configs/
|   |   |-- r024/
|   |   |   `-- full_configs/
|   |   |-- r025/
|   |   |   `-- full_configs/
|   |   |-- r026/
|   |   |   `-- full_configs/
|   |   |-- r027/
|   |   |   `-- full_configs/
|   |   |-- r028/
|   |   |   `-- full_configs/
|   |   |-- r029/
|   |   |   `-- full_configs/
|   |   |-- r030/
|   |   |   `-- full_configs/
|   |   `-- r031/
|   |       `-- full_configs/
|   |-- shared/
|   |   |-- configs/
|   |   |   |-- __init__.py
|   |   |   |-- problem_definition.yaml
|   |   |   |-- schema.py
|   |   |   `-- search_space.json
|   |   |-- eval_module.py
|   |   |-- losses.py
|   |   `-- train.py
|   |-- templates/
|   |   |-- condor_submit.template
|   |   `-- condor_wrapper.sh
|   |-- model_rounds_manifest.csv
|   `-- README.md
|-- shared_baseline/
|   |-- full_reproduction_pipeline/
|   |   |-- 01_original_data_format/
|   |   |   |-- data_formatter_fixed.py
|   |   |   `-- prepare_data.py
|   |   |-- 02_split_definition/
|   |   |   |-- concat_v2_data.py
|   |   |   `-- update_manifest_seed7.py
|   |   |-- 03_model/
|   |   |   |-- model.py
|   |   |   `-- unet_v2_baseline_adapter.py
|   |   |-- 04_training/
|   |   |   |-- losses.py
|   |   |   |-- step1_full_train.py
|   |   |   |-- train.py
|   |   |   `-- train_7level_v3.py
|   |   |-- 05_restore/
|   |   |   `-- step2_raw_restore.py
|   |   |-- 06_evaluation/
|   |   |   |-- eval_baseline_newval.py
|   |   |   |-- eval_comprehensive.py
|   |   |   |-- eval_module.py
|   |   |   `-- eval_seeds_v3.py
|   |   |-- 07_run_scripts/
|   |   |   |-- check_baseline.sh
|   |   |   |-- check_lu_split.sh
|   |   |   `-- submit_baseline_s1.sh
|   |   |-- configs/
|   |   |   |-- shared_configs/
|   |   |   |-- baseline_s1_train_config.json
|   |   |   `-- reference_seed7_train_config.json
|   |   |-- docs/
|   |   |   `-- BASELINE.md
|   |   `-- PIPELINE_OVERVIEW.md
|   |-- baseline_matrix_exact_summary.csv
|   |-- baseline_matrix_exact_summary.json
|   |-- config.json
|   |-- eval_protocol.md
|   |-- model.py
|   `-- README.md
|-- .gitignore
|-- DIRECTORY_TREE.md
|-- README.md
|-- reproducibility_notes.md
|-- requirements.txt
`-- SHA256SUMS.txt
```
