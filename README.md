# Handwriting-Research-

Implementation of the requested methodology for multi-modal handwriting analysis.

## Included components

- **Static image feature extraction** with preprocessing (binarization, artifact removal, deskew, contrast normalization), morphological measurements, and texture descriptors (LBP + GLCM).
- **Dynamic pen feature extraction** with 100 Hz interpolation, velocity/acceleration profiles, sliding-window moments, FFT spectral descriptors, and entropy metrics.
- **Multimodal integration** with KNN imputation, standardization, and PCA retaining 95% explained variance.
- **Deep learning architectures**:
  - Attention-enhanced emotion classifier (static + texture + dynamic fusion)
  - Personality regressor with mean/log-variance heads for uncertainty quantification
  - Variational Autoencoder for unsupervised representation learning
  - Conditional tabular GAN for synthetic feature generation
- **Training utilities**: ready-to-use loss functions and single-step trainers for every model type.
- **Neurocognitive interpretation mapping** to link predictive features to established neural/psychological constructs.

## Quick usage

### Feature extraction pipeline

```python
from handwriting_research.pipeline import MultiModalPipeline, HandwritingSample

pipeline = MultiModalPipeline()
# build HandwritingSample objects with image+t/x/y/pressure arrays
# reduced_features = pipeline.fit_transform(samples)
```

### Training the models

```python
import torch
from handwriting_research.models import (
    AttentionEnhancedEmotionClassifier,
    PersonalityRegressor,
    VariationalAutoencoder,
    ConditionalTabularGAN,
    GANConfig,
)
from handwriting_research.training import (
    emotion_loss,
    gaussian_nll_loss,
    vae_loss,
    gan_train_step,
)

# --- Emotion classifier (cross-entropy) ---
emotion_model = AttentionEnhancedEmotionClassifier(morph_dim=32, texture_channels=1, dynamic_dim=10, num_classes=3)
optimizer = torch.optim.Adam(emotion_model.parameters(), lr=1e-3)
logits = emotion_model(morph_batch, texture_batch, dynamic_batch)
loss = emotion_loss(logits, label_batch)
loss.backward(); optimizer.step()

# --- Personality regressor (Gaussian NLL) ---
regressor = PersonalityRegressor(input_dim=64, output_dim=5)
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
mu, logvar = regressor(feature_batch)
loss = gaussian_nll_loss(mu, logvar, target_batch)
loss.backward(); optimizer.step()

# --- Variational Autoencoder (β-ELBO) ---
vae = VariationalAutoencoder(input_dim=64, latent_dim=16)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
recon, mu, logvar = vae(feature_batch)
total, recon_loss, kl_loss = vae_loss(recon, feature_batch, mu, logvar, beta=1.0)
total.backward(); optimizer.step()

# --- Conditional tabular GAN ---
config = GANConfig(feature_dim=64, condition_dim=8)
gan = ConditionalTabularGAN(config)
opt_g = torch.optim.Adam(gan.generator.parameters(), lr=1e-4)
opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-4)
d_loss, g_loss = gan_train_step(
    gan.generator, gan.discriminator,
    real_features, condition_batch,
    config.hidden_dim, opt_g, opt_d,
)
```

## Installation

```bash
# Install the package with all runtime dependencies
pip install -e .

# Also install test dependencies (adds pytest)
pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest -q
```
