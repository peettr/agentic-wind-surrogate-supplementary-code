import torch
import torch.nn as nn


class p5_res_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.arch_name = "p5_res_unet"
        self.depth = depth

        channels = [n_c * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for out_ch in channels[:-1]:
            self.encoders.append(self._res_block(prev_channels, out_ch))
            prev_channels = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._res_block(prev_channels, channels[-1])

        self.up_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_channels = channels[-1]

        for skip_ch in reversed(channels[:-1]):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(prev_channels, skip_ch, kernel_size=3),
                    nn.BatchNorm2d(skip_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.decoders.append(self._res_block(skip_ch * 2, skip_ch))
            prev_channels = skip_ch

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_channels, out_channels, kernel_size=3),
        )

    def _res_block(self, in_ch, out_ch):
        return nn.ModuleDict(
            {
                "main": nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3),
                    nn.BatchNorm2d(out_ch),
                ),
                "skip": nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1),
                "act": nn.ReLU(inplace=True),
            }
        )

    def _apply_res_block(self, block, x):
        return block["act"](block["main"](x) + block["skip"](x))

    def forward(self, x):
        valid = ~torch.isnan(x)
        output_mask = valid.all(dim=1, keepdim=True).to(x.dtype)
        x = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        for encoder in self.encoders:
            x = self._apply_res_block(encoder, x)
            skips.append(x)
            x = self.pool(x)

        x = self._apply_res_block(self.bottleneck, x)

        for up_block, decoder, skip in zip(self.up_blocks, self.decoders, reversed(skips)):
            x = up_block(x)
            x = torch.cat((skip, x), dim=1)
            x = self._apply_res_block(decoder, x)

        x = self.out_conv(x)
        x = x * output_mask
        return x


