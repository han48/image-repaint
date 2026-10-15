"""
StyleConditionedUNet: MobileNetV3-Small encoder + U-Net decoder
with AdaIN style conditioning.
Input:  photo (B, 3, 256, 256) + style_id (B,) int
Output: (B, 3, 256, 256) RGB, sigmoid activated

Actual MobileNetV3-Small encoder feature map sizes (input 256x256):
  enc0 = features[0]        -> 16ch, 128x128  (stride 2)
  enc1 = features[1]        -> 16ch,  64x64   (stride 2)
  enc2 = features[2:4]      -> 24ch,  32x32
  enc3 = features[4:7]      -> 40ch,  16x16
  enc4 = features[7:10]     -> 96ch,   8x8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StyleConditionedUNet(nn.Module):
    """
    Encoder: MobileNetV3-Small (pretrained on ImageNet)
    Decoder: U-Net with skip connections + AdaIN style conditioning
    Supports 9 styles: 5 grayscale + 4 color (oil, pencil_color, crayon, watercolor)
    Input:  photo (B, 3, 256, 256), style_id (B,) int64
    Output: (B, 3, 256, 256) RGB with sigmoid activation
    Target: < 6M parameters
    """

    def __init__(self, num_styles: int = 9, pretrained: bool = True):
        super().__init__()

        # Style embedding: maps style_id -> 64-dim vector
        self.style_embed = nn.Embedding(num_styles, 64)

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        features = backbone.features

        # Encoder stages — slices verified by probe on 256x256 input:
        # enc0: features[0]    -> 16ch, 128x128
        # enc1: features[1]    -> 16ch,  64x64
        # enc2: features[2:4]  -> 24ch,  32x32
        # enc3: features[4:7]  -> 40ch,  16x16
        # enc4: features[7:10] -> 96ch,   8x8
        self.enc0 = features[0]                            # 16ch, 128x128
        self.enc1 = features[1]                            # 16ch,  64x64
        self.enc2 = nn.Sequential(*list(features[2:4]))    # 24ch,  32x32
        self.enc3 = nn.Sequential(*list(features[4:7]))    # 40ch,  16x16
        self.enc4 = nn.Sequential(*list(features[7:10]))   # 96ch,   8x8

        # Bridge: compress to 128ch at 8x8
        self.bridge = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Style projection layers (64-dim embedding -> C*2 for AdaIN gamma+beta)
        self.style_proj_bridge = nn.Linear(64, 128 * 2)   # 256
        self.style_proj_d4     = nn.Linear(64, 64 * 2)    # 128
        self.style_proj_d3     = nn.Linear(64, 32 * 2)    # 64
        self.style_proj_d2     = nn.Linear(64, 24 * 2)    # 48
        self.style_proj_d1     = nn.Linear(64, 16 * 2)    # 32

        # Decoder blocks (InstanceNorm2d so AdaIN works correctly):
        # dec4: up 8->16,   cat bridge(128) + e3(40) = 168ch -> 64ch
        # dec3: up 16->32,  cat d4(64)      + e2(24) =  88ch -> 32ch
        # dec2: up 32->64,  cat d3(32)      + e1(16) =  48ch -> 24ch
        # dec1: up 64->128, cat d2(24)      + e0(16) =  40ch -> 16ch
        # dec0: up 128->256, no skip        ->              16ch
        self.dec4 = self._decoder_block(128 + 40, 64)
        self.dec3 = self._decoder_block(64 + 24, 32)
        self.dec2 = self._decoder_block(32 + 16, 24)
        self.dec1 = self._decoder_block(24 + 16, 16)
        self.dec0 = self._decoder_block(16, 16)

        # Output head: 3 channel RGB, sigmoid for [0,1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def _adain(self, x: torch.Tensor, style_vec: torch.Tensor,
               proj_layer: nn.Linear) -> torch.Tensor:
        """Adaptive Instance Normalization: normalize x then scale/shift with style."""
        gb = proj_layer(style_vec)          # (B, C*2)
        C = x.shape[1]
        gamma = gb[:, :C].view(-1, C, 1, 1)
        beta  = gb[:, C:].view(-1, C, 1, 1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-5
        return gamma * (x - mean) / std + beta

    def _decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Two conv layers with InstanceNorm2d (required for AdaIN compatibility)."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        # Style embedding
        s = self.style_embed(style_id)   # (B, 64)

        # Encode
        e0 = self.enc0(x)    # 16ch, 128x128
        e1 = self.enc1(e0)   # 16ch,  64x64
        e2 = self.enc2(e1)   # 24ch,  32x32
        e3 = self.enc3(e2)   # 40ch,  16x16
        e4 = self.enc4(e3)   # 96ch,   8x8

        # Bridge
        b = self.bridge(e4)                          # 128ch, 8x8
        b = self._adain(b, s, self.style_proj_bridge)

        # Decode with skip connections + AdaIN style conditioning
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)   # 16x16
        d4 = self.dec4(torch.cat([d4, e3], dim=1))   # (128+40)->64ch, 16x16
        d4 = self._adain(d4, s, self.style_proj_d4)

        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)  # 32x32
        d3 = self.dec3(torch.cat([d3, e2], dim=1))   # (64+24)->32ch, 32x32
        d3 = self._adain(d3, s, self.style_proj_d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)  # 64x64
        d2 = self.dec2(torch.cat([d2, e1], dim=1))   # (32+16)->24ch, 64x64
        d2 = self._adain(d2, s, self.style_proj_d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)  # 128x128
        d1 = self.dec1(torch.cat([d1, e0], dim=1))   # (24+16)->16ch, 128x128
        d1 = self._adain(d1, s, self.style_proj_d1)

        # Final upsample to input resolution (128->256), no skip
        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)  # 256x256
        d0 = self.dec0(d0)                           # 16ch, 256x256

        return self.output_conv(d0)                  # 3ch, 256x256


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = StyleConditionedUNet(num_styles=9, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    style = torch.tensor([0, 5])
    y = model(x, style)
    assert y.shape == (2, 3, 256, 256), f'Wrong shape: {y.shape}'
    n = count_parameters(model)
    print(f'Output: {y.shape}, Parameters: {n:,}')
    assert n < 6_000_000, f'Too many parameters: {n:,} (limit 6M)'
    print('StyleConditionedUNet test PASSED')
