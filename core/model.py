from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvBlockGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            group_norm(out_ch),
            nn.SiLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.block(x))


class UNetPlusPlus2D(nn.Module):
    """Model 3: compact 2D U-Net++ used in the thesis experiment."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 24):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = ConvBlockGN(in_channels, nb[0], 0.0)
        self.conv1_0 = ConvBlockGN(nb[0], nb[1], 0.0)
        self.conv2_0 = ConvBlockGN(nb[1], nb[2], 0.05)
        self.conv3_0 = ConvBlockGN(nb[2], nb[3], 0.10)
        self.conv4_0 = ConvBlockGN(nb[3], nb[4], 0.10)

        self.conv0_1 = ConvBlockGN(nb[0] + nb[1], nb[0], 0.0)
        self.conv1_1 = ConvBlockGN(nb[1] + nb[2], nb[1], 0.0)
        self.conv2_1 = ConvBlockGN(nb[2] + nb[3], nb[2], 0.05)
        self.conv3_1 = ConvBlockGN(nb[3] + nb[4], nb[3], 0.10)

        self.conv0_2 = ConvBlockGN(nb[0] * 2 + nb[1], nb[0], 0.0)
        self.conv1_2 = ConvBlockGN(nb[1] * 2 + nb[2], nb[1], 0.0)
        self.conv2_2 = ConvBlockGN(nb[2] * 2 + nb[3], nb[2], 0.05)

        self.conv0_3 = ConvBlockGN(nb[0] * 3 + nb[1], nb[0], 0.0)
        self.conv1_3 = ConvBlockGN(nb[1] * 3 + nb[2], nb[1], 0.0)
        self.conv0_4 = ConvBlockGN(nb[0] * 4 + nb[1], nb[0], 0.0)
        self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

    @staticmethod
    def upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample_to(x1_0, x0_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample_to(x2_0, x1_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample_to(x3_0, x2_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample_to(x4_0, x3_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample_to(x1_1, x0_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample_to(x2_1, x1_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample_to(x3_1, x2_0)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample_to(x1_2, x0_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample_to(x2_2, x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample_to(x1_3, x0_0)], 1))
        return self.final(x0_4)


def choose_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str | Path, device: torch.device) -> Tuple[UNetPlusPlus2D, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNetPlusPlus2D(
        in_channels=int(cfg.get("in_channels", 4)),
        out_channels=int(cfg.get("out_channels", 1)),
        base=int(cfg.get("base_channels", 24)),
    ).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt
