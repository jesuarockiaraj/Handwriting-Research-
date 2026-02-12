from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .features import HandwritingFeatureExtractor


@dataclass
class LabelSchema:
    emotion: str = "emotion"
    personality: str = "personality"
    state: str = "state"


class HandwritingDataset(Dataset):
    """Dataset for multitask handwriting psychology classification.

    Expected CSV columns:
    - image_path
    - emotion
    - personality
    - state
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path = ".",
        transform: Optional[Callable] = None,
        schema: LabelSchema | None = None,
        feature_extractor: HandwritingFeatureExtractor | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.schema = schema or LabelSchema()
        self.feature_extractor = feature_extractor or HandwritingFeatureExtractor()

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((128, 512)),
                transforms.ToTensor(),
            ]
        )

        self.label_maps = self._build_label_maps()

    def _build_label_maps(self) -> Dict[str, Dict[str, int]]:
        return {
            "emotion": self._map_column(self.schema.emotion),
            "personality": self._map_column(self.schema.personality),
            "state": self._map_column(self.schema.state),
        }

    def _map_column(self, col: str) -> Dict[str, int]:
        classes = sorted(self.df[col].astype(str).unique().tolist())
        return {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        tensor = self.transform(image)
        handcrafted = torch.tensor(
            list(self.feature_extractor.extract(image).values()), dtype=torch.float32
        )

        labels = {
            "emotion": self.label_maps["emotion"][str(row[self.schema.emotion])],
            "personality": self.label_maps["personality"][str(row[self.schema.personality])],
            "state": self.label_maps["state"][str(row[self.schema.state])],
        }

        return {
            "image": tensor,
            "handcrafted": handcrafted,
            "emotion": torch.tensor(labels["emotion"], dtype=torch.long),
            "personality": torch.tensor(labels["personality"], dtype=torch.long),
            "state": torch.tensor(labels["state"], dtype=torch.long),
        }
