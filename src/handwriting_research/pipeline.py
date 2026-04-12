from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .feature_extraction import DynamicFeatureExtractor, MultimodalFeatureIntegrator, StaticFeatureExtractor


@dataclass
class HandwritingSample:
    image: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    pressure: np.ndarray


class MultiModalPipeline:
    """End-to-end feature engineering pipeline for static + dynamic handwriting."""

    def __init__(self):
        self.static_extractor = StaticFeatureExtractor()
        self.dynamic_extractor = DynamicFeatureExtractor()
        self.integrator = MultimodalFeatureIntegrator()

    def extract_sample_features(self, sample: HandwritingSample) -> Dict[str, float]:
        static_features = self.static_extractor.extract(sample.image)
        dynamic_features = self.dynamic_extractor.extract(sample.t, sample.x, sample.y, sample.pressure)
        return {**static_features, **dynamic_features}

    def fit_transform(self, samples: Iterable[HandwritingSample]) -> np.ndarray:
        rows: List[Dict[str, float]] = [self.extract_sample_features(sample) for sample in samples]
        reduced, _ = self.integrator.fit_transform(rows)
        return reduced

    def transform(self, samples: Iterable[HandwritingSample]) -> np.ndarray:
        rows: List[Dict[str, float]] = [self.extract_sample_features(sample) for sample in samples]
        return self.integrator.transform(rows)
