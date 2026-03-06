"""
🧠 AI Emotion Simulation Engine
Dynamic emotional state modeling for human-like AI behavior.
"""

from .emotion_state import EmotionState
from .emotion_engine import EmotionEngine
from .emotion_triggers import EmotionTrigger
from .emotion_transition import EmotionTransitionMatrix

__version__ = "1.0.0"
__author__ = "PranayMahendrakar"

__all__ = [
    "EmotionState",
    "EmotionEngine",
    "EmotionTrigger",
    "EmotionTransitionMatrix",
]
