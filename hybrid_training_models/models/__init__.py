"""Hybrid Model Zoo â€” 40 architectures from V3 with bug fixes.

Categories:
  baselines: UNet v2, v3, SDF-7level, AFNO hybrid
  tier_a_unet_variants: AG-UNet, NAFNet, SAC-UNet, DilatedUNet, CBAM-UNet, DCN-UNet, KAN-UNet, HRNet
  tier_b_new_architectures: Transolver, UMamba, Mamba2D, QuadMamba, HRFormer, Swin-UNETR, PerceiverIO, CNN-DeepONet
  tier_c_operators: FNO2D, AFNO, CNO, U-NO, Transolver-lite
  tier_d_hybrids: DilatedFNO, DilatedHRFormer, FNO-Encoder-Decoder, FourierUNet, HRDCN,
                  Mamba-Attention, Multiscale-Conv, Residual-Spectral, SAC-Mamba, Attention-Mamba
  helpers: Base class, AFNO block, FFNO, FNO v3, UFNO, ConvNeXt v2 UNet

Bug fixes applied (from V3 code review):
  - UMamba: SimpleSSMBlock now has real 4-direction selective SSM scan
  - SAC-UNet: SpatialAdaptiveConv now uses scale+shift modulation (was return None)
  - HRDCN: HRBlock now uses DCNv2 (was plain Conv)
  - FourierUNet: FFT pad correctly separates positive/negative frequencies
"""




