from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=-1)
        return torch.sum(seq * weights.unsqueeze(-1), dim=1)


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attn


class AttentionEnhancedEmotionClassifier(nn.Module):
    def __init__(
        self,
        morph_dim: int,
        texture_channels: int,
        dynamic_dim: int,
        num_classes: int = 3,
        lstm_hidden: int = 64,
    ):
        super().__init__()
        self.morph_branch = nn.Sequential(
            nn.Linear(morph_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.texture_conv = nn.Sequential(
            nn.Conv2d(texture_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.spatial_attention = SpatialAttention2D()
        self.texture_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic_proj = nn.Linear(dynamic_dim, 64)
        self.dynamic_lstm = nn.LSTM(64, lstm_hidden, batch_first=True, bidirectional=True)
        self.temporal_attention = TemporalAttention(2 * lstm_hidden)

        fused_dim = 64 + 32 + 2 * lstm_hidden
        self.channel_attention = ChannelAttention(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        morph_features: torch.Tensor,
        texture_features: torch.Tensor,
        dynamic_sequence: torch.Tensor,
    ) -> torch.Tensor:
        morph = self.morph_branch(morph_features)

        texture = self.texture_conv(texture_features)
        texture = self.spatial_attention(texture)
        texture = self.texture_pool(texture).flatten(1)

        dynamic = self.dynamic_proj(dynamic_sequence)
        dynamic, _ = self.dynamic_lstm(dynamic)
        dynamic = self.temporal_attention(dynamic)

        fused = torch.cat([morph, texture, dynamic], dim=1)
        fused = self.channel_attention(fused.unsqueeze(-1)).squeeze(-1)
        return self.classifier(fused)


class PersonalityRegressor(nn.Module):
    """Regression network with uncertainty estimation for personality scores."""

    def __init__(self, input_dim: int, output_dim: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, output_dim)
        self.logvar_head = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(x)
        return self.mean_head(latent), self.logvar_head(latent)


class VariationalAutoencoder(nn.Module):
    """VAE for unsupervised handwriting representation learning."""

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class GANConfig:
    feature_dim: int
    condition_dim: int
    hidden_dim: int = 128


class ConditionalTabularGAN(nn.Module):
    """Conditional GAN for feature-level augmentation."""

    def __init__(self, config: GANConfig):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(config.hidden_dim + config.condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.feature_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(config.feature_dim + config.condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.noise_dim = config.hidden_dim

    def generate(self, condition: torch.Tensor, n_samples: int) -> torch.Tensor:
        noise = torch.randn(n_samples, self.noise_dim, device=condition.device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).repeat(n_samples, 1)
        g_input = torch.cat([noise, condition], dim=1)
        return self.generator(g_input)

    def discriminate(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).repeat(features.size(0), 1)
        d_input = torch.cat([features, condition], dim=1)
        return self.discriminator(d_input)
