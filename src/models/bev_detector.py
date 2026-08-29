"""Lightweight multi-scale encoder-decoder for dense BEV detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BEVDetector(nn.Module):
    """Predict car centers and box parameters at the input BEV resolution."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBlock(3, 32)
        self.encoder_1 = ConvBlock(32, 64, stride=2)
        self.encoder_2 = ConvBlock(64, 128, stride=2)
        self.context = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder_1 = ConvBlock(128 + 64, 64)
        self.decoder_2 = ConvBlock(64 + 32, 64)

        self.heatmap_head = self._head(64, 1)
        self.offset_head = self._head(64, 2)
        self.size_head = self._head(64, 2)
        self.rotation_head = self._head(64, 2)

        # CenterNet's low foreground prior prevents a dense field of false
        # positives before the model has learned meaningful features.
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    @staticmethod
    def _head(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"inputs must have shape (B, 3, H, W), got {tuple(inputs.shape)}")
        stem = self.stem(inputs)
        encoded_1 = self.encoder_1(stem)
        encoded_2 = self.context(self.encoder_2(encoded_1))

        decoded_1 = F.interpolate(encoded_2, size=encoded_1.shape[-2:], mode="bilinear", align_corners=False)
        decoded_1 = self.decoder_1(torch.cat((decoded_1, encoded_1), dim=1))
        decoded_2 = F.interpolate(decoded_1, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        features = self.decoder_2(torch.cat((decoded_2, stem), dim=1))

        return {
            "heatmap": self.heatmap_head(features),
            "offset": self.offset_head(features),
            "size": self.size_head(features),
            "rotation": self.rotation_head(features),
        }
