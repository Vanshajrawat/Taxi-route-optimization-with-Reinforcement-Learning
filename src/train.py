"""
Training pipeline for the Q-learning Taxi agent.
"""
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.agent import QLearningAgent
from src.environment import TaxiEnvironment


@dataclass
class Hyperparameters:
    """All tunable hyperparameters for Q-learning."""
    alpha:          float = 0.1     # Learning rate
    gamma:          float = 0.99    # Discount factor
    epsilon:        float = 1.0     # Initial exploration rate
    epsilon_min:    float = 0.01    # Minimum exploration rate
    epsilon_decay:  float = 0.9995  # Per-episode multiplicative decay
    n_episodes:     int   = 3000    # Training episodes
    max_steps:      int   = 200     # Max steps per episode (env default is 200)
    random_seed:    int   = 42      # RNG seed


@dataclass
class EpisodeRecord:
    """Training log for a single episode."""
    episode:        int
    total_reward:   float
    steps:          int
    epsilon:        float
    success:        bool     # True if agent made a successful dropoff


@dataclass
class TrainingResult:
    """Aggregated output of a full training run."""
    hyperparameters:     dict
    episode_records:     List[dict]
    training_time_sec:   float
    final_epsilon:       float
    episodes_trained:    int


def train(
    hp: Optional[Hyperparameters] = None,
    verbose: bool = True,
    checkpoint_interval: int = 500,
    checkpoint_path: Optional[str] = None,
) -> tuple[QLearningAgent, TrainingResult]:
    """
    Run a full Q-learning training loop.

    Args:
        hp:                   Hyperparameters (defaults used if None).
        verbose:              Show progress bar.
        checkpoint_interval:  Save Q-table every N episodes (0 = never).
        checkpoint_path:      Where to save checkpoints.

    Returns:
        (trained_agent, training_result)
    """
    if hp is None:
        hp = Hyperparameters()

    agent = QLearningAgent(
        alpha=hp.alpha,
        gamma=hp.gamma,
        epsilon=hp.epsilon,
        epsilon_min=hp.epsilon_min,
        epsilon_decay=hp.epsilon_decay,
        random_seed=hp.random_seed,
    )
    env = TaxiEnvironment(seed=hp.random_seed)
    records: List[EpisodeRecord] = []

    start_time = time.time()
    iterator = range(hp.n_episodes)

    if verbose and HAS_TQDM:
        iterator = tqdm(iterator, desc="🚕 Training", unit="ep", ncols=80)

    for episode in iterator:
        state = env.reset()
        total_reward = 0.0
        success = False

        for step in range(hp.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Reward of +20 signals a successful dropoff
            if reward == 20:
                success = True

            if done:
                break

        agent.decay_epsilon()

        record = EpisodeRecord(
            episode=episode + 1,
            total_reward=total_reward,
            steps=step + 1,
            epsilon=round(agent.epsilon, 6),
            success=success,
        )
        records.append(record)

        # Checkpoint
        if (
            checkpoint_interval > 0
            and checkpoint_path
            and (episode + 1) % checkpoint_interval == 0
        ):
            agent.save(f"{checkpoint_path}_ep{episode + 1}.npy")

        if verbose and HAS_TQDM and (episode + 1) % 100 == 0:
            recent = records[-100:]
            avg_r = sum(r.total_reward for r in recent) / len(recent)
            sr = sum(r.success for r in recent)
            iterator.set_postfix(
                avg_r=f"{avg_r:.1f}", succ=f"{sr}%", ε=f"{agent.epsilon:.3f}"
            )

    env.close()
    elapsed = time.time() - start_time

    result = TrainingResult(
        hyperparameters=asdict(hp),
        episode_records=[asdict(r) for r in records],
        training_time_sec=round(elapsed, 2),
        final_epsilon=round(agent.epsilon, 6),
        episodes_trained=agent.episodes_trained,
    )

    if verbose:
        print(f"\n⏱  Training complete in {elapsed:.1f}s")
        print(f"   Final ε = {agent.epsilon:.4f}")
        last_100 = records[-100:]
        avg_r = sum(r.total_reward for r in last_100) / len(last_100)
        sr = sum(r.success for r in last_100)
        print(f"   Last 100 episodes → avg reward: {avg_r:.2f}, success rate: {sr}%")

    return agent, result
