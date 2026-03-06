# 🧠 AI Emotion Simulation Engine

> **Dynamic emotional state modeling for human-like AI behavior**

Not sentiment analysis. Not classification. A living, breathing **emotion state machine** that simulates how an AI agent *feels* as it processes information, encounters failures, and interacts with the world.

---

## 💡 What Makes This Different

| Traditional Approach | This Engine |
|---|---|
| Static label: "positive" / "negative" | Dynamic vector: curiosity=0.7, frustration=0.2... |
| Single emotion output | Multi-dimensional emotional space |
| No history | Temporal decay & memory |
| No interaction | Cross-emotion cascading effects |
| No training | Supervised, RL, scenario-based learning |

---

## 🧬 Core Emotional States

| Emotion | Description | Triggers |
|---|---|---|
| 🧠 **Curiosity** | Drive to explore & discover | New info, knowledge gaps, novelty |
| 😡 **Frustration** | Blocked goals, repeated failure | Task failures, contradictions, timeouts |
| 💪 **Confidence** | Certainty & self-belief | Successes, validation, user approval |
| 😕 **Uncertainty** | Ambiguity & confusion | Conflicting info, ambiguous inputs |
| ⚡ **Excitement** | High-energy positive state | Breakthroughs, discoveries |
| 😌 **Satisfaction** | Completion & fulfillment | Task completion, mastery |
| 😰 **Anxiety** | Pressure & apprehension | Deadlines, repeated failures |

---

## 🏗️ Architecture

```
ai-emotion-simulation-engine/
├── emotion_engine/
│   ├── __init__.py           # Module exports
│   ├── emotion_state.py      # EmotionVector, EmotionState, decay logic
│   ├── emotion_triggers.py   # 12 built-in triggers + TriggerLibrary
│   ├── emotion_transition.py # State machine, cascading, conflict resolution
│   └── emotion_engine.py     # Core orchestrator
├── training/
│   └── emotion_trainer.py    # Supervised, RL, scenario-based training
├── demo.py                   # 6 live demonstration scenarios
└── requirements.txt
```

---

## 🧩 How It Works

### 1. Emotion Vector
Each agent maintains a 7-dimensional emotional vector:
```python
EmotionVector(
    curiosity    = 0.7,   # 0.0 to 1.0
    frustration  = 0.1,
    confidence   = 0.6,
    uncertainty  = 0.3,
    excitement   = 0.4,
    satisfaction = 0.2,
    anxiety      = 0.0
)
```

### 2. Triggers
Events modify the emotion vector with signed deltas:
```python
trigger("task_failed")
# frustration += 0.25, confidence -= 0.20, uncertainty += 0.10
```

### 3. Cascading Effects
High-intensity emotions influence others via a **7×7 influence matrix**:
```
curiosity → amplifies excitement, reduces frustration
frustration → amplifies anxiety, suppresses confidence
confidence → suppresses uncertainty and anxiety
```

### 4. Temporal Decay
All emotions decay exponentially toward their baseline over time:
```
current = baseline + (current - baseline) × e^(-decay_rate)
```

### 5. Conflict Resolution
Emotionally contradictory states are auto-resolved:
```
high confidence + high uncertainty → uncertainty is reduced
high satisfaction + high frustration → frustration suppressed
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

### Basic Usage

```python
from emotion_engine import EmotionEngine

# Create an agent
engine = EmotionEngine(agent_id="assistant_01", verbose=True)

# Apply triggers
engine.trigger("new_information")     # curiosity increases
engine.trigger("task_failed")         # frustration rises
engine.trigger("task_succeeded")      # confidence restored

# Check current state
print(engine.current_emotion())       # CONFIDENCE
print(engine.snapshot())              # Full state dict
```

### Custom Triggers

```python
from emotion_engine.emotion_triggers import EmotionTrigger, TriggerCategory

creative_trigger = EmotionTrigger(
    name="creative_breakthrough",
    category=TriggerCategory.NOVELTY,
    deltas={"curiosity": 0.3, "excitement": 0.4, "confidence": 0.2},
    intensity_multiplier=1.2,
)
engine.add_trigger(creative_trigger)
engine.trigger("creative_breakthrough")
```

### Event Callbacks

```python
def on_change(data):
    print(f"Emotion shifted: {data['from']} → {data['to']}")

engine.on("on_dominant_change", on_change)
engine.trigger("task_failed")  # triggers callback when dominant changes
```

### Run a Simulation

```python
results = engine.simulate(
    trigger_sequence=["new_information", "task_failed", "hypothesis_confirmed"],
    steps_between=2
)
for r in results:
    print(r["dominant"], r["intensity"])
```

---

## 📊 Training the Engine

### Supervised Training
```python
from training.emotion_trainer import EmotionTrainer

trainer = EmotionTrainer(engine, learning_rate=0.01, episodes=50)

# Provide (trigger, desired_outcome) pairs
pairs = [
    ("task_succeeded", "CONFIDENCE"),
    ("new_information", "CURIOSITY"),
    ("task_failed", "FRUSTRATION"),
]
metrics = trainer.supervised_train(pairs, epochs=10)
print(metrics["final_accuracy"])
```

### Reinforcement Learning Training
```python
def reward_fn(snapshot):
    # Reward agent for being curious and confident
    return snapshot["emotions"]["curiosity"] + snapshot["emotions"]["confidence"]

metrics = trainer.reinforcement_train(reward_fn, steps_per_episode=10)
print(metrics["final_avg_reward"])
```

### Scenario-Based Training
```python
scenarios = [
    {
        "triggers": ["new_information", "task_failed", "hypothesis_confirmed"],
        "expected_final": "CONFIDENCE",
        "weight": 2.0,
    }
]
trainer.scenario_train(scenarios, epochs=5)
trainer.save_model("models/agent_01.json")
```

---

## 📋 Built-in Trigger Library

| Trigger | Category | Primary Effect |
|---|---|---|
| `new_information` | NOVELTY | +curiosity, +excitement |
| `task_failed` | FAILURE | +frustration, -confidence |
| `task_succeeded` | SUCCESS | +confidence, +satisfaction |
| `ambiguous_input` | AMBIGUITY | +uncertainty, +anxiety |
| `repeated_failure` | REPETITION | ++frustration, --confidence |
| `knowledge_gap` | KNOWLEDGE | +curiosity, +uncertainty |
| `hypothesis_confirmed` | VALIDATION | +confidence, -uncertainty |
| `contradiction_detected` | CONTRADICTION | +uncertainty, +frustration |
| `user_positive_feedback` | INTERACTION | +confidence, +satisfaction |
| `user_negative_feedback` | INTERACTION | -confidence, +frustration |
| `timeout_reached` | TIMEOUT | +frustration, +anxiety |
| `deep_exploration` | KNOWLEDGE | +confidence, +satisfaction |

---

## 📸 Demo Output

```
🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠
  AI EMOTION SIMULATION ENGINE — DEMO

Step: 3  |  Dominant: CURIOSITY  |  Intensity: 0.412
═════════════════════════════════════════════════════
  Curiosity       [█████████████████░░░░░░░░░░░░░] 0.57 ◄
  Frustration     [██████░░░░░░░░░░░░░░░░░░░░░░░░] 0.21
  Confidence      [███████████░░░░░░░░░░░░░░░░░░░] 0.38
  Uncertainty     [████████░░░░░░░░░░░░░░░░░░░░░░] 0.27
  Excitement      [███████████████░░░░░░░░░░░░░░░] 0.49
  Satisfaction    [███████░░░░░░░░░░░░░░░░░░░░░░░] 0.24
  Anxiety         [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.09
```

---

## 💡 Use Cases

- **Conversational AI** — Make chatbots emotionally responsive to user frustration or enthusiasm
- **Game NPCs** — NPCs that get frustrated when losing, excited when winning
- **Educational AI** — Agents that express uncertainty when confused, curiosity when engaged
- **Autonomous Agents** — Robots/agents with human-like behavioral adaptation
- **Research** — Study emotional dynamics in multi-agent systems

---

## 🔧 License

MIT License — Free to use, modify, and extend.

---

*Built by [PranayMahendrakar](https://github.com/PranayMahendrakar) — SONYTECH*
