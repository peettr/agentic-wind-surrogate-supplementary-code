import torch
import torch.nn as nn
import torch.nn.functional as F


class pid_dcb_unet(nn.Module):
    class _Conv3x3(nn.Module):
        def __init__(self, in_channels, out_channels, dilation=1):
            super().__init__()
            self.dilation = dilation
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                dilation=dilation,
                bias=False,
            )

        @staticmethod
        def _reflect_pad(x, padding):
            remaining = int(padding)
            while remaining > 0:
                h, w = x.shape[-2:]
                step_h = min(remaining, max(h - 1, 0))
                step_w = min(remaining, max(w - 1, 0))
                if step_h == 0 or step_w == 0:
                    return x
                x = F.pad(x, (step_w, step_w, step_h, step_h), mode="reflect")
                remaining -= min(step_h, step_w)
            return x

        def forward(self, x):
            padded = self._reflect_pad(x, self.dilation)
            if padded.shape[-2] < 2 * self.dilation + 1 or padded.shape[-1] < 2 * self.dilation + 1:
                padded = F.interpolate(
                    padded,
                    size=(
                        max(padded.shape[-2], 2 * self.dilation + 1),
                        max(padded.shape[-1], 2 * self.dilation + 1),
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                y = self.conv(padded)
                return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(padded)

    class _DCB(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            mid_channels = out_channels

            self.in_proj = nn.Sequential(
                pid_dcb_unet._Conv3x3(in_channels, mid_channels, dilation=1),
                nn.GroupNorm(min(8, mid_channels), mid_channels),
                nn.SiLU(inplace=True),
            )

            self.local = nn.Sequential(
                pid_dcb_unet._Conv3x3(mid_channels, mid_channels, dilation=1),
                nn.GroupNorm(min(8, mid_channels), mid_channels),
                nn.SiLU(inplace=True),
            )

            self.context = nn.Sequential(
                pid_dcb_unet._Conv3x3(mid_channels, mid_channels, dilation=2),
                nn.GroupNorm(min(8, mid_channels), mid_channels),
                nn.SiLU(inplace=True),
                pid_dcb_unet._Conv3x3(mid_channels, mid_channels, dilation=3),
                nn.GroupNorm(min(8, mid_channels), mid_channels),
                nn.SiLU(inplace=True),
            )

            self.mix = nn.Sequential(
                nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, padding=0, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
            )

            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            )

            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            z = self.in_proj(x)
            z_local = self.local(z)
            z_context = self.context(z)
            z = self.mix(torch.cat([z_local, z_context], dim=1))
            return self.act(z + self.skip(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self._DCB(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._DCB(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]

        for skip_channels in reversed(channels):
            self.upconvs.append(
                nn.ConvTranspose2d(decoder_in, skip_channels, kernel_size=2, stride=2, padding=0, bias=False)
            )
            self.decoders.append(self._DCB(skip_channels * 2, skip_channels))
            decoder_in = skip_channels

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        z = x_masked

        for encoder in self.encoders:
            z = encoder(z)
            skips.append(z)
            if z.shape[-2] > 1 and z.shape[-1] > 1:
                z = self.pool(z)

        z = self.bottleneck(z)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            z = upconv(z)

            if z.shape[-2:] != skip.shape[-2:]:
                z = F.interpolate(z, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            z = torch.cat([z, skip], dim=1)
            z = decoder(z)

        output = self.out_conv(self._Conv3x3._reflect_pad(z, 1))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output


