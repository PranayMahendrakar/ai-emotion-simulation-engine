"""
Emotion Triggers Module
Defines contextual events that trigger emotional state transitions.
"""

from dataclasses import dataclass
from typing import Dict, Callable, Optional, List
from enum import Enum, auto


class TriggerCategory(Enum):
    """Categories of events that can trigger emotional changes."""
    KNOWLEDGE = auto()      # Learning/discovery events
    FAILURE = auto()        # Error/failure events
    SUCCESS = auto()        # Achievement events
    AMBIGUITY = auto()      # Unclear/conflicting information
    REPETITION = auto()     # Repeated patterns
    NOVELTY = auto()        # New unexplored territory
    VALIDATION = auto()     # Confirmation of beliefs
    CONTRADICTION = auto()  # Conflicting information
    TIMEOUT = auto()        # Waiting too long
    INTERACTION = auto()    # User interaction events


@dataclass
class EmotionTrigger:
    """
    Defines an emotional trigger — an event that modifies emotional state.
    
    Each trigger has:
    - A name and category
    - Emotional deltas (how much each emotion changes)
    - A condition function (optional) for context-aware activation
    - An intensity multiplier
    """
    name: str
    category: TriggerCategory
    deltas: Dict[str, float]
    description: str = ""
    intensity_multiplier: float = 1.0
    condition: Optional[Callable] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def is_applicable(self, context: dict) -> bool:
        """Check if this trigger applies given a context dict."""
        if self.condition is None:
            return True
        try:
            return self.condition(context)
        except Exception:
            return False

    def apply(self, emotion_vector, multiplier: float = 1.0) -> None:
        """Apply this trigger's emotional deltas to an EmotionVector."""
        effective = multiplier * self.intensity_multiplier
        for emotion, delta in self.deltas.items():
            if hasattr(emotion_vector, emotion):
                current = getattr(emotion_vector, emotion)
                setattr(emotion_vector, emotion, current + delta * effective)
        emotion_vector.normalize()

    def __repr__(self) -> str:
        return f"EmotionTrigger({self.name}, cat={self.category.name})"


class TriggerLibrary:
    """
    Built-in library of common emotion triggers for AI agents.
    Covers the four primary states: curiosity, frustration, confidence, uncertainty.
    """

    @staticmethod
    def new_information() -> EmotionTrigger:
        """Encountering new, interesting information."""
        return EmotionTrigger(
            name="new_information",
            category=TriggerCategory.NOVELTY,
            deltas={"curiosity": 0.2, "excitement": 0.15, "uncertainty": 0.05},
            description="Agent encounters new, unexplored information",
            tags=["learning", "discovery"],
        )

    @staticmethod
    def task_failed() -> EmotionTrigger:
        """A task or subtask fails unexpectedly."""
        return EmotionTrigger(
            name="task_failed",
            category=TriggerCategory.FAILURE,
            deltas={"frustration": 0.25, "confidence": -0.2, "uncertainty": 0.1},
            description="Agent fails at a task or encounters an error",
            tags=["error", "failure"],
        )

    @staticmethod
    def task_succeeded() -> EmotionTrigger:
        """A task is completed successfully."""
        return EmotionTrigger(
            name="task_succeeded",
            category=TriggerCategory.SUCCESS,
            deltas={"confidence": 0.2, "satisfaction": 0.25, "frustration": -0.15},
            description="Agent successfully completes a task",
            tags=["success", "achievement"],
        )

    @staticmethod
    def ambiguous_input() -> EmotionTrigger:
        """Input or context is unclear or ambiguous."""
        return EmotionTrigger(
            name="ambiguous_input",
            category=TriggerCategory.AMBIGUITY,
            deltas={"uncertainty": 0.25, "anxiety": 0.1, "confidence": -0.1},
            description="Agent receives ambiguous or conflicting information",
            tags=["ambiguity", "confusion"],
        )

    @staticmethod
    def repeated_failure() -> EmotionTrigger:
        """Multiple consecutive failures."""
        return EmotionTrigger(
            name="repeated_failure",
            category=TriggerCategory.REPETITION,
            deltas={"frustration": 0.35, "confidence": -0.3, "anxiety": 0.15},
            description="Agent fails multiple times in a row",
            intensity_multiplier=1.5,
            tags=["failure", "repetition"],
        )

    @staticmethod
    def knowledge_gap_discovered() -> EmotionTrigger:
        """Agent discovers it lacks important knowledge."""
        return EmotionTrigger(
            name="knowledge_gap",
            category=TriggerCategory.KNOWLEDGE,
            deltas={"curiosity": 0.3, "uncertainty": 0.2, "confidence": -0.1},
            description="Agent discovers a gap in its knowledge",
            tags=["learning", "knowledge"],
        )

    @staticmethod
    def hypothesis_confirmed() -> EmotionTrigger:
        """A hypothesis or prediction is validated."""
        return EmotionTrigger(
            name="hypothesis_confirmed",
            category=TriggerCategory.VALIDATION,
            deltas={"confidence": 0.3, "satisfaction": 0.2, "uncertainty": -0.2},
            description="Agent's prediction or hypothesis is confirmed",
            tags=["validation", "confidence"],
        )

    @staticmethod
    def contradiction_detected() -> EmotionTrigger:
        """Conflicting information is discovered."""
        return EmotionTrigger(
            name="contradiction_detected",
            category=TriggerCategory.CONTRADICTION,
            deltas={"uncertainty": 0.3, "frustration": 0.15, "curiosity": 0.1},
            description="Agent detects contradictory information",
            tags=["contradiction", "conflict"],
        )

    @staticmethod
    def user_positive_feedback() -> EmotionTrigger:
        """User provides positive feedback."""
        return EmotionTrigger(
            name="user_positive_feedback",
            category=TriggerCategory.INTERACTION,
            deltas={"confidence": 0.25, "satisfaction": 0.3, "anxiety": -0.1},
            description="User provides positive feedback or approval",
            tags=["feedback", "social"],
        )

    @staticmethod
    def user_negative_feedback() -> EmotionTrigger:
        """User provides negative feedback."""
        return EmotionTrigger(
            name="user_negative_feedback",
            category=TriggerCategory.INTERACTION,
            deltas={"confidence": -0.2, "frustration": 0.2, "uncertainty": 0.15},
            description="User provides negative feedback or correction",
            tags=["feedback", "social"],
        )

    @staticmethod
    def timeout_reached() -> EmotionTrigger:
        """Operation takes too long."""
        return EmotionTrigger(
            name="timeout_reached",
            category=TriggerCategory.TIMEOUT,
            deltas={"frustration": 0.2, "anxiety": 0.15, "confidence": -0.05},
            description="Operation or waiting exceeds time threshold",
            tags=["time", "delay"],
        )

    @staticmethod
    def deep_exploration_completed() -> EmotionTrigger:
        """Agent completes deep exploration of a topic."""
        return EmotionTrigger(
            name="deep_exploration",
            category=TriggerCategory.KNOWLEDGE,
            deltas={"curiosity": -0.1, "confidence": 0.2, "satisfaction": 0.25, "uncertainty": -0.15},
            description="Agent finishes deep exploration of a knowledge domain",
            tags=["learning", "mastery"],
        )

    @classmethod
    def get_all(cls) -> Dict[str, EmotionTrigger]:
        """Return all built-in triggers as a dictionary."""
        methods = [
            cls.new_information, cls.task_failed, cls.task_succeeded,
            cls.ambiguous_input, cls.repeated_failure, cls.knowledge_gap_discovered,
            cls.hypothesis_confirmed, cls.contradiction_detected,
            cls.user_positive_feedback, cls.user_negative_feedback,
            cls.timeout_reached, cls.deep_exploration_completed,
        ]
        return {t().name: t() for t in methods}
