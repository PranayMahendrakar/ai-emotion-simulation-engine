"""
Emotion Trainer Module
Train the emotion engine using sequences of events and reward signals.
Adapts the influence matrix based on desired emotional outcomes.
"""

import json
import random
import logging
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
from pathlib import Path

from emotion_engine import EmotionEngine
from emotion_engine.emotion_triggers import TriggerLibrary, EmotionTrigger

logger = logging.getLogger(__name__)


class EmotionTrainer:
    """
    Trains the EmotionEngine to produce desired emotional responses.
    
    Training approaches:
    1. Supervised: Provide (trigger, desired_emotion) pairs
    2. Reinforcement: Reward/penalize based on emotional outcomes
    3. Scenario-based: Train on scripted interaction scenarios
    """

    def __init__(
        self, 
        engine: EmotionEngine,
        learning_rate: float = 0.01,
        episodes: int = 100,
        reward_decay: float = 0.95,
    ):
        self.engine = engine
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.reward_decay = reward_decay
        self.training_history: List[dict] = []
        self.loss_curve: List[float] = []

    def supervised_train(
        self, 
        pairs: List[Tuple[str, str]], 
        epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train on (trigger_name, target_dominant_emotion) pairs.
        
        Args:
            pairs: List of (trigger, target_emotion) tuples
            epochs: Number of training epochs
        
        Returns:
            Training metrics dict
        """
        logger.info(f"Starting supervised training: {len(pairs)} pairs, {epochs} epochs")
        
        all_losses = []
        all_accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0

            random.shuffle(pairs)

            for trigger_name, target_emotion in pairs:
                self.engine.reset()
                snapshot = self.engine.trigger(trigger_name)
                predicted = snapshot["dominant"]

                if predicted == target_emotion:
                    correct += 1
                    loss = 0.0
                else:
                    loss = 1.0
                    # Adapt the influence matrix toward desired emotion
                    self.engine.transition_matrix.adapt_influence(
                        emotion=trigger_name.split("_")[0] if "_" in trigger_name else "curiosity",
                        target=target_emotion.lower(),
                        adjustment=0.1,
                        learning_rate=self.learning_rate,
                    )

                epoch_loss += loss

            avg_loss = epoch_loss / max(len(pairs), 1)
            accuracy = correct / max(len(pairs), 1)
            all_losses.append(avg_loss)
            all_accuracies.append(accuracy)
            self.loss_curve.append(avg_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

        return {
            "losses": all_losses,
            "accuracies": all_accuracies,
            "final_accuracy": all_accuracies[-1] if all_accuracies else 0,
        }

    def scenario_train(
        self, 
        scenarios: List[Dict],
        epochs: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train on scripted scenarios.
        
        Each scenario is a dict with:
        - 'triggers': list of trigger names to apply in sequence
        - 'expected_final': expected dominant emotion at the end
        - 'weight': importance weight for this scenario (default 1.0)
        
        Args:
            scenarios: List of scenario dicts
            epochs: Training epochs
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting scenario training: {len(scenarios)} scenarios")
        
        all_losses = []
        all_accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total_weight = sum(s.get("weight", 1.0) for s in scenarios)

            for scenario in scenarios:
                triggers = scenario.get("triggers", [])
                expected = scenario.get("expected_final", "CONFIDENCE")
                weight = scenario.get("weight", 1.0)

                self.engine.reset()
                final_snapshot = None

                for t in triggers:
                    final_snapshot = self.engine.trigger(t)

                if final_snapshot:
                    predicted = final_snapshot["dominant"]
                    if predicted == expected:
                        correct += weight
                        # Positive reinforcement
                        reward = weight
                    else:
                        # Negative reinforcement
                        reward = -weight
                        for t in triggers:
                            self.engine.transition_matrix.adapt_influence(
                                emotion="confidence",
                                target=expected.lower(),
                                adjustment=0.05 * weight,
                                learning_rate=self.learning_rate,
                            )

                    epoch_loss += abs(1 - (correct / max(total_weight, 1)))

            accuracy = correct / max(total_weight, 1)
            avg_loss = epoch_loss / max(len(scenarios), 1)
            all_losses.append(avg_loss)
            all_accuracies.append(accuracy)
            self.loss_curve.append(avg_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

        return {
            "losses": all_losses,
            "accuracies": all_accuracies,
            "final_accuracy": all_accuracies[-1] if all_accuracies else 0,
        }

    def reinforcement_train(
        self, 
        reward_function,
        trigger_pool: Optional[List[str]] = None,
        steps_per_episode: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train using reinforcement learning signals.
        
        Args:
            reward_function: Callable(snapshot) -> float reward (-1 to 1)
            trigger_pool: Pool of triggers to randomly sample from
            steps_per_episode: Steps per training episode
        
        Returns:
            Training metrics
        """
        if trigger_pool is None:
            trigger_pool = self.engine.list_triggers()

        total_rewards = []
        logger.info(f"Starting RL training: {self.episodes} episodes")

        for episode in range(self.episodes):
            self.engine.reset()
            episode_reward = 0.0

            for step in range(steps_per_episode):
                # Epsilon-greedy exploration
                epsilon = max(0.1, 1.0 - (episode / self.episodes))
                
                if random.random() < epsilon:
                    trigger = random.choice(trigger_pool)
                else:
                    # Exploit: choose trigger that increased reward most recently
                    trigger = random.choice(trigger_pool[:3])

                snapshot = self.engine.trigger(trigger)
                reward = reward_function(snapshot)
                episode_reward += reward * (self.reward_decay ** step)

                # Update influence based on reward
                if reward > 0:
                    dominant = snapshot["dominant"].lower()
                    self.engine.transition_matrix.adapt_influence(
                        emotion=dominant,
                        target=dominant,
                        adjustment=reward * 0.1,
                        learning_rate=self.learning_rate,
                    )

            total_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg = sum(total_rewards[-10:]) / 10
                logger.info(f"Episode {episode+1}/{self.episodes} — avg_reward: {avg:.4f}")

        return {
            "episode_rewards": total_rewards,
            "final_avg_reward": sum(total_rewards[-10:]) / min(10, len(total_rewards)),
        }

    def save_model(self, path: str) -> None:
        """Save trained engine state and influence matrix."""
        save_data = {
            "engine_state": self.engine.to_dict(),
            "influence_matrix": self.engine.transition_matrix._influence_matrix.tolist(),
            "loss_curve": self.loss_curve,
            "metadata": {
                "learning_rate": self.learning_rate,
                "episodes": self.episodes,
                "agent_id": self.engine.agent_id,
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a previously saved model."""
        import numpy as np
        with open(path, "r") as f:
            data = json.load(f)

        # Restore influence matrix
        matrix = np.array(data["influence_matrix"])
        self.engine.transition_matrix._influence_matrix = matrix
        self.loss_curve = data.get("loss_curve", [])
        logger.info(f"Model loaded from {path}")

    def generate_training_report(self) -> str:
        """Generate a text report of training progress."""
        if not self.loss_curve:
            return "No training has been performed yet."

        initial_loss = self.loss_curve[0] if self.loss_curve else 0
        final_loss = self.loss_curve[-1] if self.loss_curve else 0
        improvement = initial_loss - final_loss

        report = [
            "=== Emotion Engine Training Report ===",
            f"Agent ID: {self.engine.agent_id}",
            f"Total training steps: {len(self.loss_curve)}",
            f"Initial loss: {initial_loss:.4f}",
            f"Final loss: {final_loss:.4f}",
            f"Improvement: {improvement:.4f} ({'+' if improvement > 0 else ''}{improvement:.1%})",
            f"Learning rate: {self.learning_rate}",
            "",
            "Current emotional state:",
            f"  Dominant emotion: {self.engine.current_emotion()}",
            f"  Intensity: {self.engine.state.vector.intensity():.3f}",
            "",
            "Trigger usage statistics:",
        ]

        for trigger, count in sorted(
            self.engine.stats["trigger_counts"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            report.append(f"  {trigger}: {count} times")

        return "\n".join(report)
