"""
Emotion Engine — Core Orchestrator
The main engine that ties together emotional states, triggers, and transitions
to simulate dynamic human-like AI emotional behavior.
"""

import time
import json
import logging
from typing import Optional, List, Dict, Callable, Any
from copy import deepcopy

from .emotion_state import EmotionState, EmotionVector, EmotionType
from .emotion_triggers import EmotionTrigger, TriggerLibrary
from .emotion_transition import EmotionTransitionMatrix

logger = logging.getLogger(__name__)


class EmotionEngine:
    """
    The central AI Emotion Simulation Engine.
    
    Orchestrates:
    - Emotional state tracking and history
    - Trigger application and cascading effects
    - State machine transitions
    - Temporal decay toward baseline
    - Callbacks on emotional changes
    - Serialization and persistence
    
    Usage:
        engine = EmotionEngine(agent_id="assistant_01")
        engine.trigger("new_information")
        print(engine.current_emotion())  # curiosity
    """

    def __init__(
        self, 
        agent_id: str = "agent",
        enable_decay: bool = True,
        decay_interval: int = 5,
        cascade_enabled: bool = True,
        verbose: bool = False,
    ):
        self.agent_id = agent_id
        self.enable_decay = enable_decay
        self.decay_interval = decay_interval
        self.cascade_enabled = cascade_enabled
        self.verbose = verbose

        # Core components
        self.state = EmotionState()
        self.transition_matrix = EmotionTransitionMatrix()

        # Trigger library
        self._trigger_library: Dict[str, EmotionTrigger] = TriggerLibrary.get_all()
        self._custom_triggers: Dict[str, EmotionTrigger] = {}

        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_trigger": [],
            "on_transition": [],
            "on_decay": [],
            "on_dominant_change": [],
        }

        # Statistics
        self.stats = {
            "total_triggers": 0,
            "total_steps": 0,
            "dominant_history": [],
            "trigger_counts": {},
        }

        self._last_dominant = self.state.vector.dominant_emotion()
        logger.info(f"EmotionEngine initialized for agent: {agent_id}")

    # ──────────────────────────────
    # Core API
    # ──────────────────────────────

    def trigger(
        self, 
        trigger_name: str, 
        context: Optional[dict] = None,
        intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Apply a named trigger to modify the emotional state.
        
        Args:
            trigger_name: Name of the trigger (built-in or custom)
            context: Optional context dict for conditional triggers
            intensity: Multiplier for trigger intensity (0.0 to 2.0)
        
        Returns:
            Current emotional state snapshot
        """
        ctx = context or {}

        # Find trigger
        trigger = (
            self._trigger_library.get(trigger_name) or
            self._custom_triggers.get(trigger_name)
        )

        if trigger is None:
            logger.warning(f"Unknown trigger: {trigger_name}")
            return self.snapshot()

        if not trigger.is_applicable(ctx):
            logger.debug(f"Trigger {trigger_name} not applicable in context")
            return self.snapshot()

        # Save previous state
        prev_vector = deepcopy(self.state.vector)
        prev_dominant = self._last_dominant

        # Apply trigger
        trigger.apply(self.state.vector, multiplier=intensity)

        # Apply cascading effects
        if self.cascade_enabled and self.transition_matrix.should_trigger_cascade(self.state.vector):
            self.state.vector = self.transition_matrix.compute_cascading_effects(self.state.vector)

        # Resolve emotional conflicts
        self.state.vector = self.transition_matrix.resolve_conflicts(self.state.vector)

        # Apply decay periodically
        if self.enable_decay and self.stats["total_steps"] % self.decay_interval == 0:
            self.state.apply_decay()
            self._fire_callbacks("on_decay", self.state.snapshot())

        # Update metadata
        self.state.step += 1
        self.state.timestamp = time.time()
        self.state.context = trigger_name
        self.state.record()

        # Track dominant emotion changes
        new_dominant = self.state.vector.dominant_emotion()
        if new_dominant != prev_dominant:
            self._last_dominant = new_dominant
            self.transition_matrix.log_transition(
                step=self.state.step,
                from_dominant=prev_dominant.name,
                to_dominant=new_dominant.name,
                trigger=trigger_name,
            )
            self._fire_callbacks("on_dominant_change", {
                "from": prev_dominant.name,
                "to": new_dominant.name,
                "step": self.state.step,
            })

        # Update stats
        self.stats["total_triggers"] += 1
        self.stats["total_steps"] += 1
        self.stats["trigger_counts"][trigger_name] = (
            self.stats["trigger_counts"].get(trigger_name, 0) + 1
        )

        # Fire trigger callbacks
        snapshot = self.snapshot()
        self._fire_callbacks("on_trigger", {
            "trigger": trigger_name,
            "snapshot": snapshot,
        })

        if self.verbose:
            dominant = snapshot["dominant"]
            intensity_val = snapshot["intensity"]
            print(f"[{self.agent_id}] Step {self.state.step}: {trigger_name} → dominant={dominant} (intensity={intensity_val:.3f})")

        return snapshot

    def step(self, context: str = "") -> Dict[str, Any]:
        """
        Advance one time step (apply decay without a trigger).
        Useful for idle simulation.
        """
        if self.enable_decay:
            self.state.apply_decay()

        if self.cascade_enabled:
            self.state.vector = self.transition_matrix.compute_cascading_effects(
                self.state.vector, learning_rate=0.05
            )

        self.state.step += 1
        self.state.timestamp = time.time()
        self.state.context = context or f"idle_step_{self.state.step}"
        self.state.record()
        self.stats["total_steps"] += 1

        return self.snapshot()

    def current_emotion(self) -> str:
        """Return the name of the current dominant emotion."""
        return self.state.vector.dominant_emotion().name

    def get_emotion_value(self, emotion: str) -> float:
        """Get current value of a specific emotion (0.0 to 1.0)."""
        return getattr(self.state.vector, emotion, 0.0)

    def snapshot(self) -> Dict[str, Any]:
        """Return a full snapshot of the current emotional state."""
        snap = self.state.snapshot()
        snap["agent_id"] = self.agent_id
        return snap

    def reset(self) -> None:
        """Reset the engine to baseline emotional state."""
        self.state.reset()
        self._last_dominant = self.state.vector.dominant_emotion()
        self.stats = {
            "total_triggers": 0,
            "total_steps": 0,
            "dominant_history": [],
            "trigger_counts": {},
        }
        logger.info(f"EmotionEngine reset for agent: {self.agent_id}")

    # ──────────────────────────────
    # Trigger Management
    # ──────────────────────────────

    def add_trigger(self, trigger: EmotionTrigger) -> None:
        """Register a custom emotion trigger."""
        self._custom_triggers[trigger.name] = trigger
        logger.debug(f"Custom trigger added: {trigger.name}")

    def remove_trigger(self, trigger_name: str) -> bool:
        """Remove a custom trigger by name."""
        if trigger_name in self._custom_triggers:
            del self._custom_triggers[trigger_name]
            return True
        return False

    def list_triggers(self) -> List[str]:
        """List all available trigger names."""
        return list(self._trigger_library.keys()) + list(self._custom_triggers.keys())

    # ──────────────────────────────
    # Callbacks
    # ──────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for an emotion event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _fire_callbacks(self, event: str, data: Any) -> None:
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error on {event}: {e}")

    # ──────────────────────────────
    # Analysis
    # ──────────────────────────────

    def emotion_summary(self) -> Dict[str, Any]:
        """Return a full summary of the engine's emotional history."""
        return {
            "agent_id": self.agent_id,
            "current_state": self.snapshot(),
            "total_steps": self.stats["total_steps"],
            "total_triggers": self.stats["total_triggers"],
            "most_triggered": sorted(
                self.stats["trigger_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "dominant_transitions": self.transition_matrix.get_transition_summary(),
            "history_length": len(self.state.history),
        }

    def get_history(self) -> List[dict]:
        """Return the full emotional history."""
        return self.state.history.copy()

    def simulate(
        self, 
        trigger_sequence: List[str], 
        steps_between: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Run a simulation with a sequence of triggers.
        
        Args:
            trigger_sequence: List of trigger names to apply
            steps_between: Idle steps between triggers
        
        Returns:
            List of state snapshots after each trigger
        """
        results = []
        for trigger_name in trigger_sequence:
            # Apply idle steps
            for _ in range(steps_between):
                self.step()
            # Apply trigger
            snapshot = self.trigger(trigger_name)
            results.append(snapshot)
        return results

    # ──────────────────────────────
    # Serialization
    # ──────────────────────────────

    def to_dict(self) -> dict:
        """Serialize engine state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "step": self.state.step,
            "vector": self.state.vector.as_dict(),
            "context": self.state.context,
            "stats": self.stats,
            "transition_log": self.transition_matrix.transition_log[-20:],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize engine state to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionEngine":
        """Restore engine from a dictionary."""
        engine = cls(agent_id=data.get("agent_id", "agent"))
        vec_data = data.get("vector", {})
        engine.state.vector = EmotionVector(**vec_data)
        engine.state.step = data.get("step", 0)
        engine.state.context = data.get("context", "")
        engine.stats = data.get("stats", engine.stats)
        return engine

    def __repr__(self) -> str:
        dominant = self.current_emotion()
        return (
            f"EmotionEngine(agent={self.agent_id}, "
            f"step={self.state.step}, dominant={dominant})"
        )
