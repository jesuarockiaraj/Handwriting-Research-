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
- **Neurocognitive interpretation mapping** to link predictive features to established neural/psychological constructs.

## Quick usage

```python
from handwriting_research.pipeline import MultiModalPipeline, HandwritingSample

pipeline = MultiModalPipeline()
# build HandwritingSample objects with image+t/x/y/pressure arrays
# reduced_features = pipeline.fit_transform(samples)
```

## Run tests

```bash
python -m pytest -q
```
