"""
Emotion State Module
Defines the core emotional states and their properties.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional
import time
import math


class EmotionType(Enum):
    """Primary emotion types supported by the engine."""
    CURIOSITY = auto()
    FRUSTRATION = auto()
    CONFIDENCE = auto()
    UNCERTAINTY = auto()
    NEUTRAL = auto()
    EXCITEMENT = auto()
    SATISFACTION = auto()
    ANXIETY = auto()


@dataclass
class EmotionVector:
    """
    Represents the multi-dimensional emotional state as a vector.
    Each dimension is normalized between 0.0 and 1.0.
    """
    curiosity: float = 0.3
    frustration: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.2
    excitement: float = 0.1
    satisfaction: float = 0.3
    anxiety: float = 0.0

    def normalize(self) -> None:
        """Clamp all values between 0.0 and 1.0."""
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.frustration = max(0.0, min(1.0, self.frustration))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.uncertainty = max(0.0, min(1.0, self.uncertainty))
        self.excitement = max(0.0, min(1.0, self.excitement))
        self.satisfaction = max(0.0, min(1.0, self.satisfaction))
        self.anxiety = max(0.0, min(1.0, self.anxiety))

    def dominant_emotion(self) -> EmotionType:
        """Return the dominant (highest intensity) emotion."""
        emotion_map = {
            EmotionType.CURIOSITY: self.curiosity,
            EmotionType.FRUSTRATION: self.frustration,
            EmotionType.CONFIDENCE: self.confidence,
            EmotionType.UNCERTAINTY: self.uncertainty,
            EmotionType.EXCITEMENT: self.excitement,
            EmotionType.SATISFACTION: self.satisfaction,
            EmotionType.ANXIETY: self.anxiety,
        }
        return max(emotion_map, key=emotion_map.get)

    def as_dict(self) -> Dict[str, float]:
        """Return emotions as a dictionary."""
        return {
            "curiosity": self.curiosity,
            "frustration": self.frustration,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "excitement": self.excitement,
            "satisfaction": self.satisfaction,
            "anxiety": self.anxiety,
        }

    def intensity(self) -> float:
        """Overall emotional intensity (average of all states)."""
        vals = [
            self.curiosity, self.frustration, self.confidence,
            self.uncertainty, self.excitement, self.satisfaction, self.anxiety
        ]
        return sum(vals) / len(vals)


@dataclass
class EmotionState:
    """
    Full emotional state snapshot of the AI agent at a point in time.
    Tracks history, decay rates, and contextual metadata.
    """
    vector: EmotionVector = field(default_factory=EmotionVector)
    timestamp: float = field(default_factory=time.time)
    context: str = ""
    step: int = 0

    # Decay rates per emotion (higher = faster decay toward neutral)
    decay_rates: Dict[str, float] = field(default_factory=lambda: {
        "curiosity": 0.05,
        "frustration": 0.08,
        "confidence": 0.02,
        "uncertainty": 0.06,
        "excitement": 0.10,
        "satisfaction": 0.03,
        "anxiety": 0.07,
    })

    # History of emotional states
    history: list = field(default_factory=list)

    def apply_decay(self) -> None:
        """Apply temporal decay toward baseline neutral values."""
        baseline = {
            "curiosity": 0.3,
            "frustration": 0.0,
            "confidence": 0.5,
            "uncertainty": 0.2,
            "excitement": 0.1,
            "satisfaction": 0.3,
            "anxiety": 0.0,
        }
        for emotion, rate in self.decay_rates.items():
            current = getattr(self.vector, emotion)
            base = baseline[emotion]
            # Exponential decay toward baseline
            new_val = base + (current - base) * math.exp(-rate)
            setattr(self.vector, emotion, new_val)
        self.vector.normalize()

    def snapshot(self) -> dict:
        """Create a snapshot of the current emotional state."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "context": self.context,
            "emotions": self.vector.as_dict(),
            "dominant": self.vector.dominant_emotion().name,
            "intensity": round(self.vector.intensity(), 3),
        }

    def record(self) -> None:
        """Save current state to history."""
        self.history.append(self.snapshot())
        if len(self.history) > 100:
            self.history.pop(0)

    def reset(self) -> None:
        """Reset to neutral baseline state."""
        self.vector = EmotionVector()
        self.step = 0
        self.context = ""

    def __repr__(self) -> str:
        dominant = self.vector.dominant_emotion().name
        intensity = round(self.vector.intensity(), 3)
        return f"EmotionState(step={self.step}, dominant={dominant}, intensity={intensity})"
