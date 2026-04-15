import numpy as np
import pytest
import torch

from handwriting_research.feature_extraction import DynamicFeatureExtractor, MultimodalFeatureIntegrator, StaticFeatureExtractor
from handwriting_research.models import AttentionEnhancedEmotionClassifier, ConditionalTabularGAN, GANConfig, PersonalityRegressor, VariationalAutoencoder
from handwriting_research.pipeline import HandwritingSample, MultiModalPipeline
from handwriting_research.training import emotion_loss, gan_train_step, gaussian_nll_loss, vae_loss


def synthetic_image(size=32, seed=0):
    rng = np.random.default_rng(seed)
    img = np.ones((size, size), dtype=np.float32)
    img[10:22, 8:24] = 0.2
    img[14:16, 8:24] = 0.0
    img += rng.normal(0, 0.08, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def synthetic_trace(n=120, seed=0):
    rng = np.random.default_rng(seed)
    amplitude = 1.0 + rng.uniform(-0.4, 0.4)
    t = np.linspace(0, 1.2, n)
    x = amplitude * np.sin(2 * np.pi * t) + 0.1 * t
    y = amplitude * np.cos(2 * np.pi * t)
    p = 0.5 + 0.2 * np.sin(4 * np.pi * t)
    return t, x, y, p


def test_static_and_dynamic_extractors_produce_features():
    static = StaticFeatureExtractor()
    dynamic = DynamicFeatureExtractor()

    sf = static.extract(synthetic_image(seed=0))
    t, x, y, p = synthetic_trace(seed=0)
    df = dynamic.extract(t, x, y, p)

    assert "slant_angle" in sf
    assert "glcm_entropy" in sf
    assert "speed_entropy" in df
    assert "pressure_spectral_energy" in df


def test_multimodal_pipeline_dimensionality_reduction():
    pipeline = MultiModalPipeline()
    samples = []
    for i in range(8):
        t, x, y, p = synthetic_trace(seed=i)
        sample = HandwritingSample(
            image=synthetic_image(seed=i),
            t=t,
            x=x,
            y=y,
            pressure=p,
        )
        samples.append(sample)

    transformed = pipeline.fit_transform(samples)
    assert transformed.shape[0] == len(samples)
    assert transformed.shape[1] > 0


def test_model_forward_shapes():
    emotion_model = AttentionEnhancedEmotionClassifier(morph_dim=16, texture_channels=1, dynamic_dim=10, num_classes=3)
    morph = torch.randn(4, 16)
    texture = torch.randn(4, 1, 16, 16)
    dynamic = torch.randn(4, 50, 10)
    logits = emotion_model(morph, texture, dynamic)
    assert logits.shape == (4, 3)

    regressor = PersonalityRegressor(input_dim=32, output_dim=5)
    mu, logvar = regressor(torch.randn(4, 32))
    assert mu.shape == (4, 5)
    assert logvar.shape == (4, 5)

    vae = VariationalAutoencoder(input_dim=32, latent_dim=8)
    recon, z_mu, z_logvar = vae(torch.randn(4, 32))
    assert recon.shape == (4, 32)
    assert z_mu.shape == (4, 8)
    assert z_logvar.shape == (4, 8)


def test_integrator_handles_missing_values():
    rows = [
        {"a": 1.0, "b": np.nan, "c": 0.3},
        {"a": 2.0, "b": 1.0, "c": np.nan},
        {"a": 3.0, "b": 2.0, "c": 0.9},
        {"a": 4.0, "b": 3.0, "c": 1.0},
        {"a": 5.0, "b": 5.0, "c": 1.2},
        {"a": 6.0, "b": 8.0, "c": 1.8},
    ]
    integrator = MultimodalFeatureIntegrator(explained_variance=0.95, n_neighbors=2)
    reduced, keys = integrator.fit_transform(rows)
    assert reduced.shape[0] == len(rows)
    assert "a" in keys and "b" in keys and "c" in keys


def test_vae_loss_components():
    vae = VariationalAutoencoder(input_dim=32, latent_dim=8)
    x = torch.randn(4, 32)
    recon, mu, logvar = vae(x)
    total, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=1.0)
    assert total.shape == ()
    assert recon_loss.item() >= 0
    # KL divergence can be negative per-sample but mean is >= -0.5*latent_dim
    assert total.item() == pytest.approx(recon_loss.item() + kl_loss.item(), rel=1e-5)


def test_gaussian_nll_loss_shape():
    regressor = PersonalityRegressor(input_dim=32, output_dim=5)
    x = torch.randn(4, 32)
    mu, logvar = regressor(x)
    target = torch.randn(4, 5)
    loss = gaussian_nll_loss(mu, logvar, target)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_emotion_loss_shape():
    emotion_model = AttentionEnhancedEmotionClassifier(morph_dim=16, texture_channels=1, dynamic_dim=10, num_classes=3)
    morph = torch.randn(4, 16)
    texture = torch.randn(4, 1, 16, 16)
    dynamic = torch.randn(4, 50, 10)
    logits = emotion_model(morph, texture, dynamic)
    labels = torch.randint(0, 3, (4,))
    loss = emotion_loss(logits, labels)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_gan_train_step_losses():
    config = GANConfig(feature_dim=16, condition_dim=4, hidden_dim=32)
    gan = ConditionalTabularGAN(config)
    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-3)
    real = torch.randn(8, 16)
    cond = torch.randn(8, 4)
    d_loss, g_loss = gan_train_step(gan.generator, gan.discriminator, real, cond, config.hidden_dim, opt_g, opt_d)
    assert d_loss.shape == ()
    assert g_loss.shape == ()
    assert not torch.isnan(d_loss)
    assert not torch.isnan(g_loss)
