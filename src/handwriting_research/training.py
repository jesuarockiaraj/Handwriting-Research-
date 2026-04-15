"""Training utilities: loss functions and single-step trainers for all model types."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evidence Lower Bound (ELBO) loss for the VariationalAutoencoder.

    Uses mean-squared-error reconstruction loss and the analytical KL divergence
    between the approximate posterior N(mu, exp(logvar)) and the standard normal N(0, I).

    Args:
        recon: Reconstructed feature vector output by the decoder, shape (B, D).
        x: Original input feature vector, shape (B, D).
        mu: Latent mean, shape (B, L).
        logvar: Latent log-variance, shape (B, L).
        beta: Weight applied to the KL term (β-VAE formulation). Default: 1.0.

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss), all scalar tensors.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def gaussian_nll_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss for the PersonalityRegressor.

    Matches the parametric output (mu, logvar) to a target regression value under a
    diagonal Gaussian likelihood.  Minimising this loss jointly learns the predictive
    mean and its uncertainty.

    Args:
        mu: Predicted mean, shape (B, K).
        logvar: Predicted log-variance, shape (B, K).
        target: Ground-truth regression targets, shape (B, K).

    Returns:
        Scalar loss tensor.
    """
    return torch.mean(0.5 * (logvar + (target - mu).pow(2) / (logvar.exp() + 1e-8)))


def emotion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss for the AttentionEnhancedEmotionClassifier.

    Args:
        logits: Raw class scores, shape (B, C).
        labels: Integer class indices, shape (B,).
        class_weights: Optional per-class weights tensor of shape (C,) for handling
            class imbalance.  Passed directly to ``F.cross_entropy``.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, labels, weight=class_weights)


def gan_train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real_features: torch.Tensor,
    condition: torch.Tensor,
    noise_dim: int,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single GAN training step for the ConditionalTabularGAN.

    Performs one discriminator update followed by one generator update using the
    standard minimax (binary cross-entropy) objective.

    Args:
        generator: The generator sub-network (``ConditionalTabularGAN.generator``).
        discriminator: The discriminator sub-network (``ConditionalTabularGAN.discriminator``).
        real_features: Batch of real feature vectors, shape (B, feature_dim).
        condition: Conditioning vector, shape (B, condition_dim) or (condition_dim,).
        noise_dim: Dimensionality of the latent noise fed to the generator.
        optimizer_g: Optimizer for the generator parameters.
        optimizer_d: Optimizer for the discriminator parameters.

    Returns:
        Tuple ``(d_loss, g_loss)`` — discriminator and generator scalar losses.
    """
    batch_size = real_features.size(0)
    device = real_features.device

    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    if condition.dim() == 1:
        condition = condition.unsqueeze(0).expand(batch_size, -1)

    # --- Discriminator step ---
    optimizer_d.zero_grad()
    noise = torch.randn(batch_size, noise_dim, device=device)
    fake_features = generator(torch.cat([noise, condition], dim=1)).detach()
    d_real = discriminator(torch.cat([real_features, condition], dim=1))
    d_fake = discriminator(torch.cat([fake_features, condition], dim=1))
    d_loss = F.binary_cross_entropy(d_real, real_labels) + F.binary_cross_entropy(d_fake, fake_labels)
    d_loss.backward()
    optimizer_d.step()

    # --- Generator step ---
    optimizer_g.zero_grad()
    noise = torch.randn(batch_size, noise_dim, device=device)
    fake_features = generator(torch.cat([noise, condition], dim=1))
    d_fake_for_g = discriminator(torch.cat([fake_features, condition], dim=1))
    g_loss = F.binary_cross_entropy(d_fake_for_g, real_labels)
    g_loss.backward()
    optimizer_g.step()

    return d_loss.detach(), g_loss.detach()
