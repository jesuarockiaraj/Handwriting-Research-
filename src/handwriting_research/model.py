from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class ModelConfig:
    emotion_classes: int
    personality_classes: int
    state_classes: int
    handcrafted_dim: int = 21
    cnn_channels: int = 128
    rnn_hidden: int = 128
    attention_heads: int = 4
    dropout: float = 0.2


class ConvStem(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntegratedHandwritingModel(nn.Module):
    """CNN + BiGRU + attention multitask model.

    Produces simultaneous predictions for:
    - emotional valence category
    - personality profile category
    - psychological state category
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.cnn = ConvStem(config.cnn_channels)
        self.rnn = nn.GRU(
            input_size=config.cnn_channels * 32,
            hidden_size=config.rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=config.rnn_hidden * 2,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        fusion_dim = config.rnn_hidden * 2 + config.handcrafted_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        self.emotion_head = nn.Linear(256, config.emotion_classes)
        self.personality_head = nn.Linear(256, config.personality_classes)
        self.state_head = nn.Linear(256, config.state_classes)

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.cnn(image)
        b, c, h, w = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        seq, _ = self.rnn(x)

        attended, _ = self.attn(seq, seq, seq)
        pooled = attended.mean(dim=1)

        fused = torch.cat([pooled, handcrafted], dim=1)
        latent = self.fusion(fused)

        return {
            "emotion": self.emotion_head(latent),
            "personality": self.personality_head(latent),
            "state": self.state_head(latent),
        }
