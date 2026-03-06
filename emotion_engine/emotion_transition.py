"""
Emotion Transition Module
Defines the state machine and transition rules between emotional states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .emotion_state import EmotionType, EmotionVector


class EmotionTransitionMatrix:
    """
    Probabilistic transition matrix for emotional state changes.
    
    Models how emotions influence each other over time.
    Based on Markov chain principles applied to multi-dimensional emotional space.
    """

    def __init__(self):
        # Cross-emotion influence matrix
        # Row: source emotion, Column: target emotion
        # Value: influence coefficient (-1 to 1)
        self.emotions = [
            "curiosity", "frustration", "confidence",
            "uncertainty", "excitement", "satisfaction", "anxiety"
        ]

        # Influence matrix: how each emotion affects others
        # Positive = amplifies, Negative = suppresses
        self._influence_matrix = np.array([
            #  cur    frus   conf   uncer  excit  satis  anx
            [  0.10,  -0.05,  0.10,  0.05,  0.20,  0.05, -0.05],  # curiosity
            [ -0.10,   0.15, -0.20,  0.15, -0.10, -0.15,  0.20],  # frustration
            [  0.05,  -0.15,  0.10, -0.20,  0.10,  0.15, -0.15],  # confidence
            [  0.10,   0.10, -0.15,  0.10, -0.05, -0.10,  0.15],  # uncertainty
            [  0.15,  -0.05,  0.05, -0.05,  0.10,  0.10, -0.05],  # excitement
            [  0.05,  -0.20,  0.15, -0.15,  0.05,  0.10, -0.15],  # satisfaction
            [ -0.10,   0.15, -0.20,  0.20, -0.10, -0.10,  0.10],  # anxiety
        ], dtype=float)

        # Transition thresholds: minimum intensity to trigger cascade
        self.transition_thresholds: Dict[str, float] = {
            "curiosity": 0.6,
            "frustration": 0.7,
            "confidence": 0.7,
            "uncertainty": 0.65,
            "excitement": 0.75,
            "satisfaction": 0.7,
            "anxiety": 0.65,
        }

        # History of transitions
        self.transition_log: List[dict] = []

    def compute_cascading_effects(
        self, 
        vector: EmotionVector,
        learning_rate: float = 0.1
    ) -> EmotionVector:
        """
        Compute how current emotional states influence each other.
        Returns a new EmotionVector with cascading effects applied.
        """
        current = np.array([
            vector.curiosity, vector.frustration, vector.confidence,
            vector.uncertainty, vector.excitement, vector.satisfaction, vector.anxiety
        ])

        # Compute influence: each emotion affects all others
        influences = np.dot(self._influence_matrix.T, current)

        # Apply only when above threshold
        deltas = influences * learning_rate

        # Create new vector with cascading effects
        new_vals = current + deltas
        new_vals = np.clip(new_vals, 0.0, 1.0)

        new_vector = EmotionVector(
            curiosity=float(new_vals[0]),
            frustration=float(new_vals[1]),
            confidence=float(new_vals[2]),
            uncertainty=float(new_vals[3]),
            excitement=float(new_vals[4]),
            satisfaction=float(new_vals[5]),
            anxiety=float(new_vals[6]),
        )
        return new_vector

    def detect_transitions(
        self, 
        prev_vector: EmotionVector,
        curr_vector: EmotionVector
    ) -> List[Tuple[str, str, float]]:
        """
        Detect significant emotional state transitions.
        Returns list of (emotion, direction, magnitude) tuples.
        """
        transitions = []
        prev_dict = prev_vector.as_dict()
        curr_dict = curr_vector.as_dict()

        for emotion in self.emotions:
            prev_val = prev_dict[emotion]
            curr_val = curr_dict[emotion]
            delta = curr_val - prev_val

            if abs(delta) >= 0.1:  # Significant change threshold
                direction = "UP" if delta > 0 else "DOWN"
                transitions.append((emotion, direction, round(abs(delta), 3)))

        return transitions

    def should_trigger_cascade(self, vector: EmotionVector) -> bool:
        """Check if any emotion is above threshold for cascading."""
        vec_dict = vector.as_dict()
        for emotion, threshold in self.transition_thresholds.items():
            if vec_dict.get(emotion, 0) >= threshold:
                return True
        return False

    def resolve_conflicts(self, vector: EmotionVector) -> EmotionVector:
        """
        Resolve emotionally conflicting states.
        E.g., high confidence + high uncertainty → reduce uncertainty.
        """
        # Confidence and uncertainty are inversely related
        if vector.confidence > 0.7 and vector.uncertainty > 0.5:
            vector.uncertainty *= 0.7

        # Satisfaction suppresses frustration
        if vector.satisfaction > 0.6 and vector.frustration > 0.4:
            vector.frustration *= 0.6

        # Anxiety and confidence suppress each other
        if vector.anxiety > 0.6 and vector.confidence > 0.5:
            vector.confidence *= 0.85
            vector.anxiety *= 0.85

        # High curiosity can coexist with uncertainty (exploration mindset)
        if vector.curiosity > 0.7 and vector.uncertainty > 0.4:
            vector.uncertainty = min(vector.uncertainty, 0.5)

        vector.normalize()
        return vector

    def get_dominant_pair(self, vector: EmotionVector) -> Tuple[str, str]:
        """Get the top two dominant emotions."""
        vals = vector.as_dict()
        sorted_emotions = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[0][0], sorted_emotions[1][0]

    def log_transition(
        self, 
        step: int, 
        from_dominant: str, 
        to_dominant: str, 
        trigger: Optional[str] = None
    ) -> None:
        """Log a state transition for analysis."""
        self.transition_log.append({
            "step": step,
            "from": from_dominant,
            "to": to_dominant,
            "trigger": trigger,
        })
        if len(self.transition_log) > 200:
            self.transition_log.pop(0)

    def get_transition_summary(self) -> Dict[str, int]:
        """Summarize transitions as a count of (from, to) pairs."""
        summary = {}
        for log in self.transition_log:
            key = f"{log['from']} -> {log['to']}"
            summary[key] = summary.get(key, 0) + 1
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    def adapt_influence(
        self, 
        emotion: str, 
        target: str, 
        adjustment: float,
        learning_rate: float = 0.01
    ) -> None:
        """
        Adapt the influence matrix based on observed patterns.
        Enables learning of personalized emotional dynamics.
        """
        if emotion in self.emotions and target in self.emotions:
            i = self.emotions.index(emotion)
            j = self.emotions.index(target)
            self._influence_matrix[i][j] += adjustment * learning_rate
            # Keep in valid range
            self._influence_matrix[i][j] = np.clip(
                self._influence_matrix[i][j], -1.0, 1.0
            )

    def __repr__(self) -> str:
        return f"EmotionTransitionMatrix(emotions={len(self.emotions)}, logged_transitions={len(self.transition_log)})"
