"""
🧠 AI Emotion Simulation Engine — Demo Script
Demonstrates the dynamic emotional state simulation system.

Run:
    python demo.py
"""

import sys
import time
sys.path.insert(0, ".")

from emotion_engine import EmotionEngine
from emotion_engine.emotion_triggers import EmotionTrigger, TriggerCategory


def print_bar(value: float, label: str, width: int = 30) -> str:
    """Helper to render a text-based emotion bar."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<15} [{bar}] {value:.2f}"


def display_state(engine: EmotionEngine) -> None:
    """Pretty-print the current emotional state."""
    snap = engine.snapshot()
    emotions = snap["emotions"]
    dominant = snap["dominant"]
    intensity = snap["intensity"]
    step = snap["step"]

    print(f"\n{'='*55}")
    print(f"  Step: {step}  |  Dominant: {dominant}  |  Intensity: {intensity:.3f}")
    print(f"{'='*55}")
    for emotion, value in emotions.items():
        bar = print_bar(value, emotion.capitalize())
        marker = " ◄" if emotion == dominant.lower() else ""
        print(bar + marker)
    print()


def demo_basic() -> None:
    """Demo 1: Basic trigger sequence."""
    print("\n" + "━"*55)
    print("  DEMO 1: Basic Emotion Triggers")
    print("━"*55)

    engine = EmotionEngine(agent_id="basic_demo", verbose=True)
    print("\n[Initial state — neutral baseline]")
    display_state(engine)

    print("[Trigger: new_information — agent discovers something new]")
    engine.trigger("new_information")
    display_state(engine)

    print("[Trigger: task_failed — agent encounters an error]")
    engine.trigger("task_failed")
    display_state(engine)

    print("[Trigger: ambiguous_input — confusing information]")
    engine.trigger("ambiguous_input")
    display_state(engine)

    print("[Trigger: task_succeeded — agent completes task successfully]")
    engine.trigger("task_succeeded")
    display_state(engine)

    print("[Trigger: user_positive_feedback — user approves]")
    engine.trigger("user_positive_feedback")
    display_state(engine)


def demo_repeated_failure() -> None:
    """Demo 2: Cascading frustration from repeated failures."""
    print("\n" + "━"*55)
    print("  DEMO 2: Cascading Frustration — Repeated Failures")
    print("━"*55)

    engine = EmotionEngine(agent_id="frustrated_agent", verbose=True)

    print("\n[Applying 5 consecutive failures...]")
    for i in range(5):
        engine.trigger("task_failed")
        print(f"  After failure #{i+1}: {engine.current_emotion()}")

    print("\n[Trying repeated_failure trigger...]")
    engine.trigger("repeated_failure")
    display_state(engine)

    print("[Recovery: hypothesis_confirmed]")
    engine.trigger("hypothesis_confirmed")
    display_state(engine)

    print("[Recovery: user_positive_feedback]")
    engine.trigger("user_positive_feedback")
    display_state(engine)


def demo_curiosity_driven() -> None:
    """Demo 3: Curiosity-driven exploration sequence."""
    print("\n" + "━"*55)
    print("  DEMO 3: Curiosity-Driven Exploration")
    print("━"*55)

    engine = EmotionEngine(agent_id="curious_agent", verbose=True)

    sequence = [
        "new_information",
        "knowledge_gap",
        "new_information",
        "contradiction_detected",
        "hypothesis_confirmed",
        "deep_exploration",
        "task_succeeded",
    ]

    print("\nRunning exploration sequence...")
    for trigger in sequence:
        time.sleep(0.1)
        engine.trigger(trigger)

    display_state(engine)
    
    summary = engine.emotion_summary()
    print("\n📊 Emotion Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Total triggers: {summary['total_triggers']}")
    print(f"  Top triggers: {summary['most_triggered'][:3]}")
    print(f"  Transitions: {summary['dominant_transitions']}")


def demo_simulation() -> None:
    """Demo 4: Automated simulation."""
    print("\n" + "━"*55)
    print("  DEMO 4: Automated Simulation (10 steps)")
    print("━"*55)

    engine = EmotionEngine(agent_id="simulation_agent", verbose=True)

    scenario = [
        "new_information", "task_failed", "ambiguous_input",
        "user_negative_feedback", "repeated_failure",
        "hypothesis_confirmed", "task_succeeded",
        "user_positive_feedback", "deep_exploration", "task_succeeded",
    ]

    results = engine.simulate(scenario, steps_between=2)

    print("\n📈 Simulation Results:")
    print(f"{'Step':<6} {'Trigger':<28} {'Dominant':<20} {'Intensity'}")
    print("-"*70)
    for i, (trigger, result) in enumerate(zip(scenario, results)):
        print(f"{result['step']:<6} {trigger:<28} {result['dominant']:<20} {result['intensity']:.3f}")

    print("\n" + "━"*55)
    print(f"  Final state: {engine.current_emotion()}")


def demo_custom_trigger() -> None:
    """Demo 5: Custom trigger creation."""
    print("\n" + "━"*55)
    print("  DEMO 5: Custom Emotion Triggers")
    print("━"*55)

    engine = EmotionEngine(agent_id="custom_agent", verbose=True)

    # Define a custom trigger for "creative breakthrough"
    creative_trigger = EmotionTrigger(
        name="creative_breakthrough",
        category=TriggerCategory.NOVELTY,
        deltas={
            "curiosity": 0.3,
            "excitement": 0.4,
            "confidence": 0.2,
            "uncertainty": -0.1,
            "anxiety": -0.15,
        },
        description="Agent achieves a creative insight or breakthrough",
        intensity_multiplier=1.2,
        tags=["creativity", "insight"],
    )

    # Define a "deadline_pressure" trigger
    deadline_trigger = EmotionTrigger(
        name="deadline_pressure",
        category=TriggerCategory.TIMEOUT,
        deltas={
            "anxiety": 0.35,
            "frustration": 0.15,
            "confidence": -0.1,
            "excitement": -0.05,
        },
        description="Agent is under time pressure",
        intensity_multiplier=1.0,
        tags=["pressure", "time"],
    )

    engine.add_trigger(creative_trigger)
    engine.add_trigger(deadline_trigger)

    print(f"\n  Available triggers: {len(engine.list_triggers())}")
    print("\n[Applying custom triggers...]")
    
    engine.trigger("deadline_pressure")
    display_state(engine)

    engine.trigger("creative_breakthrough")
    display_state(engine)


def demo_callbacks() -> None:
    """Demo 6: Event callbacks."""
    print("\n" + "━"*55)
    print("  DEMO 6: Event-Driven Callbacks")
    print("━"*55)

    engine = EmotionEngine(agent_id="callback_agent")

    # Register callbacks
    transitions_seen = []

    def on_dominant_change(data):
        transitions_seen.append(f"{data['from']} → {data['to']} (step {data['step']})")
        print(f"  🔄 TRANSITION: {data['from']} → {data['to']}")

    def on_trigger(data):
        trigger = data["trigger"]
        dominant = data["snapshot"]["dominant"]
        print(f"  ⚡ TRIGGER: {trigger} → {dominant}")

    engine.on("on_dominant_change", on_dominant_change)
    engine.on("on_trigger", on_trigger)

    triggers = ["task_failed", "task_failed", "repeated_failure", "task_succeeded", 
                "user_positive_feedback", "deep_exploration"]

    print("\nApplying trigger sequence with callbacks:\n")
    for t in triggers:
        engine.trigger(t)

    print(f"\n  Total dominant transitions: {len(transitions_seen)}")
    for t in transitions_seen:
        print(f"    {t}")


if __name__ == "__main__":
    print("\n" + "🧠" * 10)
    print("  AI EMOTION SIMULATION ENGINE — DEMO")
    print("🧠" * 10)

    demo_basic()
    demo_repeated_failure()
    demo_curiosity_driven()
    demo_simulation()
    demo_custom_trigger()
    demo_callbacks()

    print("\n" + "="*55)
    print("  ✅ All demos completed successfully!")
    print("="*55 + "\n")
