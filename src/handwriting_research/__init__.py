"""Handwriting research toolkit implementing multi-modal feature extraction and learning models."""

from .feature_extraction import (
    DynamicFeatureExtractor,
    MultimodalFeatureIntegrator,
    StaticFeatureExtractor,
)
from .interpretation import NeurocognitiveInterpreter
from .models import (
    AttentionEnhancedEmotionClassifier,
    ConditionalTabularGAN,
    GANConfig,
    PersonalityRegressor,
    VariationalAutoencoder,
)
from .training import (
    emotion_loss,
    gan_train_step,
    gaussian_nll_loss,
    vae_loss,
)

__all__ = [
    "StaticFeatureExtractor",
    "DynamicFeatureExtractor",
    "MultimodalFeatureIntegrator",
    "AttentionEnhancedEmotionClassifier",
    "PersonalityRegressor",
    "VariationalAutoencoder",
    "ConditionalTabularGAN",
    "GANConfig",
    "NeurocognitiveInterpreter",
    "vae_loss",
    "gaussian_nll_loss",
    "emotion_loss",
    "gan_train_step",
]
