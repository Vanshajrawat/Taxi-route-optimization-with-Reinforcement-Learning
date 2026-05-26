"""
Evaluation module: runs the trained policy on unseen episodes
and collects detailed performance metrics.
"""
from dataclasses import dataclass, asdict
from typing import List

from src.agent import QLearningAgent
from src.environment import TaxiEnvironment


@dataclass
class StepRecord:
    step: int
    state: int
    state_decoded: str
    action: int
    action_name: str
    reward: float
    next_state: int
    done: bool


@dataclass
class EvalEpisode:
    episode: int
    total_reward: float
    steps: int
    success: bool
    steps_log: List[dict]   # Only populated when full_trace=True


@dataclass
class EvalResult:
    n_episodes:      int
    avg_reward:      float
    std_reward:      float
    min_reward:      float
    max_reward:      float
    avg_steps:       float
    success_rate:    float
    episodes:        List[dict]


def evaluate(
    agent: QLearningAgent,
    n_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    full_trace: bool = False,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate a trained Q-learning agent (greedy policy, no exploration).

    Args:
        agent:       Trained QLearningAgent.
        n_episodes:  Number of test episodes.
        max_steps:   Step limit per episode.
        seed:        Env seed (different from training seed).
        full_trace:  If True, record every step in each episode.
        verbose:     Print summary to stdout.

    Returns:
        EvalResult with aggregated metrics and per-episode data.
    """
    import numpy as np

    env = TaxiEnvironment(seed=seed)
    episode_records: List[EvalEpisode] = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        success = False
        steps_log: List[dict] = []

        for step in range(max_steps):
            action = agent.choose_action(state, greedy=True)
            next_state, reward, done, _ = env.step(action)

            if full_trace:
                steps_log.append(asdict(StepRecord(
                    step=step + 1,
                    state=state,
                    state_decoded=TaxiEnvironment.decode_state_str(state),
                    action=action,
                    action_name=QLearningAgent.ACTION_NAMES[action],
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )))

            total_reward += reward
            state = next_state

            if reward == 20:
                success = True
            if done:
                break

        episode_records.append(EvalEpisode(
            episode=ep + 1,
            total_reward=total_reward,
            steps=step + 1,
            success=success,
            steps_log=steps_log,
        ))

    env.close()

    rewards = np.array([r.total_reward for r in episode_records])
    steps   = np.array([r.steps       for r in episode_records])

    result = EvalResult(
        n_episodes=n_episodes,
        avg_reward=round(float(rewards.mean()), 4),
        std_reward=round(float(rewards.std()),  4),
        min_reward=round(float(rewards.min()),  4),
        max_reward=round(float(rewards.max()),  4),
        avg_steps=round(float(steps.mean()),    4),
        success_rate=round(
            sum(r.success for r in episode_records) / n_episodes * 100, 2
        ),
        episodes=[asdict(r) for r in episode_records],
    )

    if verbose:
        print("\n" + "=" * 50)
        print("📊  EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Episodes      : {result.n_episodes}")
        print(f"  Avg Reward    : {result.avg_reward:+.2f}  (±{result.std_reward:.2f})")
        print(f"  Min / Max     : {result.min_reward:+.0f} / {result.max_reward:+.0f}")
        print(f"  Avg Steps     : {result.avg_steps:.1f}")
        print(f"  Success Rate  : {result.success_rate:.1f}%")
        print("=" * 50)

    return result
