from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn


DEFAULT_LETTERSTR = "abcdefghjklmnopqrstuvwxyz"


@dataclass(frozen=True)
class CaptchaConfig:
    letterstr: str = DEFAULT_LETTERSTR
    captcha_len: int = 4
    image_height: int = 100
    image_width: int = 120

    @property
    def num_classes(self) -> int:
        return len(self.letterstr)


class CaptchaCNN(nn.Module):
    def __init__(self, config: CaptchaConfig):
        super().__init__()
        self.config = config

        # Simple, robust backbone; AdaptiveAvgPool2d removes shape fragility.
        self.backbone = nn.Sequential(
            _conv_block(3, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            _conv_block(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            _conv_block(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            _conv_block(128, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList(
            [nn.Linear(256, config.num_classes) for _ in range(config.captcha_len)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (N, 3, H, W)
        features = self.backbone(x)
        features = features.flatten(1)
        features = self.dropout(features)
        return [head(features) for head in self.heads]

    @torch.inference_mode()
    def predict_text(self, x: torch.Tensor) -> List[str]:
        self.eval()
        logits_list = self.forward(x)
        preds: List[torch.Tensor] = [torch.argmax(l, dim=1) for l in logits_list]
        # preds is list of (N,), convert to N strings
        batch = x.shape[0]
        results: List[str] = []
        for i in range(batch):
            chars = [self.config.letterstr[int(p[i].item())] for p in preds]
            results.append("".join(chars))
        return results


def _conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def decode_indices(letterstr: str, indices: Sequence[int]) -> str:
    return "".join(letterstr[i] for i in indices)


def encode_text(letterstr: str, text: str) -> List[int]:
    indices: List[int] = []
    for ch in text:
        idx = letterstr.find(ch)
        if idx < 0:
            raise ValueError(f"Unsupported character: {ch!r}")
        indices.append(idx)
    return indices
