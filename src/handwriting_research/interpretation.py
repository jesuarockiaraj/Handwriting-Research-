from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class NeurocognitiveInterpreter:
    """Maps predictive features to plausible neural and psychological constructs."""

    emotion_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "speed_entropy": ["Limbic flexibility", "Anterior cingulate conflict monitoring"],
            "pressure_entropy": ["Arousal regulation", "Ventromedial prefrontal modulation"],
            "slant_angle": ["Affective motor output", "Emotion-linked motor planning"],
            "glcm_entropy": ["Fine-motor variability", "Cortico-striatal signal diversity"],
        }
    )
    personality_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "mean_speed": ["Extraversion-related psychomotor tempo"],
            "std_pressure": ["Neuroticism-linked autonomic variability"],
            "mean_interchar_spacing": ["Conscientiousness and planning consistency"],
            "baseline_strength": ["Executive control persistence"],
        }
    )

    def interpret(self, ranked_features: Iterable[str], target: str = "emotion") -> Dict[str, List[str]]:
        mapping = self.emotion_map if target == "emotion" else self.personality_map
        return {feature: mapping.get(feature, ["No established mapping; exploratory signal"]) for feature in ranked_features}
