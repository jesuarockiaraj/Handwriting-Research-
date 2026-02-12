from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image


@dataclass
class FeatureConfig:
    """Configuration for handcrafted feature extraction."""

    bins: int = 16
    edge_threshold: float = 0.2


class HandwritingFeatureExtractor:
    """Extract global and local descriptors from handwriting images.

    The extractor returns a compact set of interpretable signals that can be
    paired with neural embeddings to improve robustness and interpretability.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def extract(self, image: Image.Image) -> Dict[str, float]:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        features = {
            "ink_density": float(1.0 - gray.mean()),
            "contrast": float(gray.std()),
            "slant_proxy": self._estimate_slant(gray),
            "line_variability": self._line_variability(gray),
            "edge_density": self._edge_density(gray),
        }

        hist = self._intensity_histogram(gray)
        for idx, value in enumerate(hist):
            features[f"hist_{idx}"] = float(value)

        return features

    def _intensity_histogram(self, gray: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(gray, bins=self.config.bins, range=(0.0, 1.0), density=True)
        return hist.astype(np.float32)

    def _estimate_slant(self, gray: np.ndarray) -> float:
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        angle = np.arctan2(gy, gx + 1e-6)
        return float(np.mean(np.sin(angle)))

    def _line_variability(self, gray: np.ndarray) -> float:
        row_energy = (1.0 - gray).mean(axis=1)
        return float(np.std(row_energy))

    def _edge_density(self, gray: np.ndarray) -> float:
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge_map = np.sqrt(gx * gx + gy * gy)
        return float((edge_map > self.config.edge_threshold).mean())
