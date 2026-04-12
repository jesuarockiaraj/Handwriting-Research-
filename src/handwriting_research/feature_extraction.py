from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class StaticFeatureExtractor:
    """Extracts morphological and texture features from static handwriting images."""

    lbp_radius: int = 1
    lbp_points: int = 8

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = self._contrast_normalize(image)
        binary = self._binarize(image)
        binary = self._remove_artifacts(binary)
        binary = self._deskew(binary)
        return binary

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        processed = self.preprocess(image)
        morph = self._morphological_features(processed)
        texture = self._texture_features(processed)
        return {**morph, **texture}

    def _contrast_normalize(self, image: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(image, [2, 98])
        denom = max(p98 - p2, 1e-6)
        return np.clip((image - p2) / denom, 0, 1)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        threshold = np.mean(image)
        return (image < threshold).astype(np.uint8)

    def _remove_artifacts(self, binary: np.ndarray) -> np.ndarray:
        cleaned = ndimage.binary_opening(binary, structure=np.ones((2, 2))).astype(np.uint8)
        return medfilt(cleaned, kernel_size=3).astype(np.uint8)

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary > 0)
        if len(xs) < 2:
            return binary
        coords = np.column_stack((xs, ys)).astype(np.float32)
        coords -= coords.mean(axis=0)
        cov = np.cov(coords, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]
        angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
        return ndimage.rotate(binary, -angle, reshape=False, order=0)

    def _morphological_features(self, binary: np.ndarray) -> Dict[str, float]:
        row_sums = binary.sum(axis=1)
        col_sums = binary.sum(axis=0)
        active_rows = np.where(row_sums > 0)[0]
        active_cols = np.where(col_sums > 0)[0]
        letter_height = float(active_rows[-1] - active_rows[0] + 1) if len(active_rows) else 0.0

        baseline_row = int(np.argmax(row_sums)) if np.any(row_sums) else 0
        baseline_strength = float(row_sums[baseline_row]) if np.any(row_sums) else 0.0

        ys, xs = np.where(binary > 0)
        slant = 0.0
        if len(xs) > 1:
            centered = np.column_stack((xs, ys)).astype(np.float32)
            centered -= centered.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            slant = float(np.degrees(np.arctan2(vh[0, 1], vh[0, 0])))

        spacing = np.diff(active_cols) if len(active_cols) > 1 else np.array([0])
        return {
            "letter_height": letter_height,
            "baseline_row": float(baseline_row),
            "baseline_strength": baseline_strength,
            "slant_angle": slant,
            "mean_interchar_spacing": float(np.mean(spacing)),
            "std_interchar_spacing": float(np.std(spacing)),
        }

    def _texture_features(self, binary: np.ndarray) -> Dict[str, float]:
        lbp_hist = self._lbp_histogram(binary)
        glcm = self._glcm(binary)
        features = {f"lbp_{i}": float(v) for i, v in enumerate(lbp_hist)}
        features.update(self._glcm_features(glcm))
        return features

    def _lbp_histogram(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        codes = []
        for y in range(self.lbp_radius, h - self.lbp_radius):
            for x in range(self.lbp_radius, w - self.lbp_radius):
                center = image[y, x]
                neighbors = []
                for p in range(self.lbp_points):
                    angle = 2 * np.pi * p / self.lbp_points
                    nx = int(round(x + self.lbp_radius * np.cos(angle)))
                    ny = int(round(y - self.lbp_radius * np.sin(angle)))
                    neighbors.append(1 if image[ny, nx] >= center else 0)
                code = int("".join(map(str, neighbors)), 2)
                codes.append(code)
        hist, _ = np.histogram(codes, bins=2**self.lbp_points, range=(0, 2**self.lbp_points), density=True)
        return hist if np.any(hist) else np.zeros(2**self.lbp_points)

    def _glcm(self, image: np.ndarray, distance: int = 1, levels: int = 2) -> np.ndarray:
        glcm = np.zeros((levels, levels), dtype=np.float32)
        h, w = image.shape
        for y in range(h):
            for x in range(w - distance):
                i = int(image[y, x])
                j = int(image[y, x + distance])
                glcm[i, j] += 1
        glcm_sum = glcm.sum()
        return glcm / glcm_sum if glcm_sum else glcm

    def _glcm_features(self, glcm: np.ndarray) -> Dict[str, float]:
        i, j = np.indices(glcm.shape)
        contrast = np.sum(glcm * (i - j) ** 2)
        homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
        energy = np.sum(glcm**2)
        glcm_entropy = -np.sum(glcm * np.log2(glcm + 1e-12))
        return {
            "glcm_contrast": float(contrast),
            "glcm_homogeneity": float(homogeneity),
            "glcm_energy": float(energy),
            "glcm_entropy": float(glcm_entropy),
        }


@dataclass
class DynamicFeatureExtractor:
    """Extracts motion and pressure based features from dynamic pen traces."""

    target_hz: int = 100
    window_size: int = 20

    def extract(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, pressure: np.ndarray) -> Dict[str, float]:
        t_u, x_u, y_u, p_u = self._interpolate(t, x, y, pressure)
        vx, vy, speed, ax, ay, accel = self._kinematics(t_u, x_u, y_u)
        win_feats = self._windowed_moments(speed, p_u)
        spectral = self._spectral_features(t_u, speed, p_u)
        entropy_feats = {
            "speed_entropy": float(self._signal_entropy(speed)),
            "pressure_entropy": float(self._signal_entropy(p_u)),
        }

        return {
            "mean_speed": float(speed.mean()),
            "std_speed": float(speed.std()),
            "mean_accel": float(accel.mean()),
            "std_accel": float(accel.std()),
            "mean_pressure": float(p_u.mean()),
            "std_pressure": float(p_u.std()),
            **win_feats,
            **spectral,
            **entropy_feats,
        }

    def _interpolate(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, pressure: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = t.astype(np.float64)
        start, stop = t.min(), t.max()
        uniform_t = np.arange(start, stop, 1.0 / self.target_hz)
        kind = "linear"
        fx = interp1d(t, x, kind=kind, bounds_error=False, fill_value="extrapolate")
        fy = interp1d(t, y, kind=kind, bounds_error=False, fill_value="extrapolate")
        fp = interp1d(t, pressure, kind=kind, bounds_error=False, fill_value="extrapolate")
        return uniform_t, fx(uniform_t), fy(uniform_t), fp(uniform_t)

    def _kinematics(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        dt = np.gradient(t)
        vx = np.gradient(x) / (dt + 1e-9)
        vy = np.gradient(y) / (dt + 1e-9)
        speed = np.sqrt(vx**2 + vy**2)
        ax = np.gradient(vx) / (dt + 1e-9)
        ay = np.gradient(vy) / (dt + 1e-9)
        accel = np.sqrt(ax**2 + ay**2)
        return vx, vy, speed, ax, ay, accel

    def _windowed_moments(self, speed: np.ndarray, pressure: np.ndarray) -> Dict[str, float]:
        stats = {}
        for name, signal in (("speed", speed), ("pressure", pressure)):
            means, variances = [], []
            for i in range(0, len(signal) - self.window_size + 1, self.window_size):
                window = signal[i : i + self.window_size]
                means.append(np.mean(window))
                variances.append(np.var(window))
            stats[f"window_mean_{name}"] = float(np.mean(means) if means else 0.0)
            stats[f"window_var_{name}"] = float(np.mean(variances) if variances else 0.0)
        return stats

    def _spectral_features(self, t: np.ndarray, speed: np.ndarray, pressure: np.ndarray) -> Dict[str, float]:
        dt = np.mean(np.diff(t))
        fs = 1.0 / max(dt, 1e-9)
        f = rfftfreq(len(speed), d=1 / fs)
        speed_fft = np.abs(rfft(speed))
        pressure_fft = np.abs(rfft(pressure))

        return {
            "speed_spectral_centroid": float(np.sum(f * speed_fft) / (np.sum(speed_fft) + 1e-9)),
            "pressure_spectral_centroid": float(np.sum(f * pressure_fft) / (np.sum(pressure_fft) + 1e-9)),
            "speed_spectral_energy": float(np.sum(speed_fft**2) / len(speed_fft)),
            "pressure_spectral_energy": float(np.sum(pressure_fft**2) / len(pressure_fft)),
        }

    def _signal_entropy(self, signal: np.ndarray, bins: int = 32) -> float:
        hist, _ = np.histogram(signal, bins=bins, density=True)
        return float(entropy(hist + 1e-12, base=2))


@dataclass
class MultimodalFeatureIntegrator:
    """Integrates static and dynamic features with standardization, imputation and PCA."""

    explained_variance: float = 0.95
    n_neighbors: int = 5

    def __post_init__(self) -> None:
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.explained_variance)

    def fit_transform(self, feature_rows: Iterable[Dict[str, float]]) -> Tuple[np.ndarray, Tuple[str, ...]]:
        matrix, keys = self._dicts_to_matrix(feature_rows)
        matrix = self.imputer.fit_transform(matrix)
        matrix = self.scaler.fit_transform(matrix)
        reduced = self.pca.fit_transform(matrix)
        return reduced, keys

    def transform(self, feature_rows: Iterable[Dict[str, float]]) -> np.ndarray:
        matrix, _ = self._dicts_to_matrix(feature_rows)
        matrix = self.imputer.transform(matrix)
        matrix = self.scaler.transform(matrix)
        return self.pca.transform(matrix)

    def _dicts_to_matrix(self, rows: Iterable[Dict[str, float]]) -> Tuple[np.ndarray, Tuple[str, ...]]:
        rows = list(rows)
        if not rows:
            raise ValueError("feature_rows cannot be empty")
        keys = tuple(sorted({k for row in rows for k in row.keys()}))
        matrix = np.array([[row.get(key, np.nan) for key in keys] for row in rows], dtype=np.float32)
        return matrix, keys
