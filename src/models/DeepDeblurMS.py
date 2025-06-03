import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Residual Block
# ---------------------------------------
class ResBlock(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feats, num_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feats, num_feats, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

# ---------------------------------------
# Single-Scale Deblurring Network
# ---------------------------------------
class SingleScaleDeblurNet(nn.Module):
    def __init__(self, in_channels, num_feats=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(num_feats) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(num_feats, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return out

# ---------------------------------------
# Multi-Scale Deblurring Network (DeepDeblurMS)
# ---------------------------------------
class DeepDeblurMS(nn.Module):
    def __init__(self):
        super().__init__()
        # Each stage expects concatenated inputs → 6 channels: [blur, upsampled_output]
        self.coarse_net = SingleScaleDeblurNet(in_channels=6)
        self.middle_net = SingleScaleDeblurNet(in_channels=6)
        self.fine_net = SingleScaleDeblurNet(in_channels=6)

    def forward(self, x):
        # Create image pyramid
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x_half, scale_factor=0.5, mode='bilinear', align_corners=False)

        # Coarse scale input: duplicate x_quarter to simulate [blur, blur]
        coarse_input = torch.cat([x_quarter, x_quarter], dim=1)
        coarse_out = self.coarse_net(coarse_input)
        up_coarse = F.interpolate(coarse_out, scale_factor=2, mode='bilinear', align_corners=False)

        # Middle scale input: [blur_half, upsampled_coarse]
        mid_input = torch.cat([x_half, up_coarse], dim=1)
        mid_out = self.middle_net(mid_input)
        up_mid = F.interpolate(mid_out, scale_factor=2, mode='bilinear', align_corners=False)

        # Fine scale input: [blur_full, upsampled_middle]
        fine_input = torch.cat([x, up_mid], dim=1)
        fine_out = self.fine_net(fine_input)

        return fine_out, mid_out, coarse_out

