import torch
import torch.nn as nn
import torch.nn.functional as F


class _SafeReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            if len(padding) != 4:
                raise ValueError("padding must be an int or a 4-tuple")
            self.padding = tuple(int(p) for p in padding)

    @staticmethod
    def _indices(size, before, after, device):
        idx = torch.arange(-before, size + after, device=device)
        if size <= 1:
            return torch.zeros_like(idx, dtype=torch.long)
        period = 2 * (size - 1)
        idx = torch.remainder(idx, period)
        idx = torch.where(idx < size, idx, period - idx)
        return idx.long()

    def forward(self, x):
        left, right, top, bottom = self.padding
        if left == right == top == bottom == 0:
            return x
        h_idx = self._indices(x.shape[-2], top, bottom, x.device)
        w_idx = self._indices(x.shape[-1], left, right, x.device)
        return x.index_select(-2, h_idx).index_select(-1, w_idx)


class adaptive_local_implicit_expert_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = min(8, out_ch)
            while out_ch % groups != 0:
                groups -= 1
            self.net = nn.Sequential(
                _SafeReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                _SafeReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _ExpertBlock(nn.Module):
        def __init__(self, ch, n_experts=4):
            super().__init__()
            self.n_experts = n_experts
            self.gate = nn.Sequential(
                _SafeReflectionPad2d(1),
                nn.Conv2d(ch, n_experts, 3),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    _SafeReflectionPad2d(d),
                    nn.Conv2d(ch, ch, 3, dilation=d, bias=False),
                    nn.GroupNorm(8 if ch % 8 == 0 else 4 if ch % 4 == 0 else 1, ch),
                    nn.SiLU(inplace=True),
                    _SafeReflectionPad2d(1),
                    nn.Conv2d(ch, ch, 3, bias=False),
                )
                for d in (1, 2, 3, 4)[:n_experts]
            ])
            self.norm = nn.GroupNorm(8 if ch % 8 == 0 else 4 if ch % 4 == 0 else 1, ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            weights = torch.softmax(self.gate(x), dim=1)
            y = 0
            for i, expert in enumerate(self.experts):
                y = y + expert(x) * weights[:, i:i + 1]
            return self.act(self.norm(x + y))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(2)

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            self._ConvBlock(bottleneck_ch, bottleneck_ch),
            self._ExpertBlock(bottleneck_ch),
            self._ExpertBlock(bottleneck_ch),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(channels[:-1]):
            self.up_convs.append(nn.Sequential(
                _SafeReflectionPad2d(1),
                nn.Conv2d(prev_ch, skip_ch, 3, bias=False),
                nn.GroupNorm(8 if skip_ch % 8 == 0 else 4 if skip_ch % 4 == 0 else 1, skip_ch),
                nn.SiLU(inplace=True),
            ))
            self.decoders.append(self._ConvBlock(skip_ch * 2, skip_ch))
            prev_ch = skip_ch

        self.local_refine = nn.Sequential(
            self._ExpertBlock(channels[0]),
            _SafeReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(8 if channels[0] % 8 == 0 else 4 if channels[0] % 4 == 0 else 1, channels[0]),
            nn.SiLU(inplace=True),
        )

        self.head = nn.Sequential(
            _SafeReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            _SafeReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, skips):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        h = self.local_refine(h)
        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)
        else:
            valid_out = valid

        output = output.clone()
        output[~valid_out] = float("nan")
        return output