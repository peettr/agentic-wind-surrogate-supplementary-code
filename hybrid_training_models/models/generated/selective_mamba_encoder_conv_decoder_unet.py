import torch
import torch.nn as nn
import torch.nn.functional as F

class selective_mamba_encoder_conv_decoder_unet(nn.Module):
    class ReflectConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                selective_mamba_encoder_conv_decoder_unet.ReflectConv2d(in_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                selective_mamba_encoder_conv_decoder_unet.ReflectConv2d(out_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class SelectiveMamba2DBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm = nn.GroupNorm(min(8, channels), channels)
            self.in_proj = nn.Conv2d(channels, channels * 2, 1, padding=0, bias=False)
            self.dw_h = nn.Conv1d(channels, channels, 7, padding=3, groups=channels, bias=False)
            self.dw_w = nn.Conv1d(channels, channels, 7, padding=3, groups=channels, bias=False)
            self.gate = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
            self.out_proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            v, g = self.in_proj(x).chunk(2, dim=1)
            b, c, h, w = v.shape

            h_seq = v.mean(dim=3)
            h_seq = self.dw_h(h_seq).unsqueeze(3).expand(-1, -1, -1, w)

            w_seq = v.mean(dim=2)
            w_seq = self.dw_w(w_seq).unsqueeze(2).expand(-1, -1, h, -1)

            selective = torch.sigmoid(self.gate(v))
            y = selective * (h_seq + w_seq) + (1.0 - selective) * v
            y = self.out_proj(y * torch.sigmoid(g))
            return residual + y

    class EncoderStage(nn.Module):
        def __init__(self, in_channels, out_channels, use_pool=True):
            super().__init__()
            self.pool = nn.AvgPool2d(2) if use_pool else nn.Identity()
            self.conv = selective_mamba_encoder_conv_decoder_unet.ConvBlock(in_channels, out_channels)
            self.mamba = selective_mamba_encoder_conv_decoder_unet.SelectiveMamba2DBlock(out_channels)

        def forward(self, x):
            x = self.pool(x)
            x = self.conv(x)
            x = self.mamba(x)
            return x

    class DecoderStage(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.conv = selective_mamba_encoder_conv_decoder_unet.ConvBlock(
                in_channels + skip_channels, out_channels
            )

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i, ch in enumerate(channels):
            self.encoders.append(
                self.EncoderStage(prev_channels, ch, use_pool=(i != 0))
            )
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.SelectiveMamba2DBlock(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.decoders.append(
                self.DecoderStage(decoder_in, skip_ch, skip_ch)
            )
            decoder_in = skip_ch

        self.head = nn.Sequential(
            self.ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for encoder in self.encoders:
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y


