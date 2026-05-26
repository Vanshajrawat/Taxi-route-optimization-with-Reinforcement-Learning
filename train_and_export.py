"""
One-click entry point: train agent, evaluate, generate plots, export JSON.

Usage:
    python train_and_export.py
    python train_and_export.py --episodes 5000 --alpha 0.15 --gamma 0.99
"""
import io
import sys
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import json
import argparse

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.train     import train, Hyperparameters
from src.evaluate  import evaluate
from src.visualize import generate_all_plots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")


def parse_args():
    p = argparse.ArgumentParser(description="Train Taxi Q-Learning Agent")
    p.add_argument("--episodes",       type=int,   default=5000,   help="Training episodes")
    p.add_argument("--alpha",          type=float, default=0.1,    help="Learning rate")
    p.add_argument("--gamma",          type=float, default=0.99,   help="Discount factor")
    p.add_argument("--epsilon",        type=float, default=1.0,    help="Initial epsilon")
    p.add_argument("--epsilon-min",    type=float, default=0.01,   help="Minimum epsilon")
    p.add_argument("--epsilon-decay",  type=float, default=0.9992, help="Epsilon decay rate")
    p.add_argument("--seed",           type=int,   default=42,     help="Random seed")
    p.add_argument("--eval-episodes",  type=int,   default=100,    help="Evaluation episodes")
    p.add_argument("--no-plots",       action="store_true",        help="Skip plot generation")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  🚕  TAXI ROUTE OPTIMIZATION — Q-LEARNING AGENT")
    print("=" * 60)

    hp = Hyperparameters(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        n_episodes=args.episodes,
        random_seed=args.seed,
    )

    # ── 1. Train ──────────────────────────────────────────────────────
    agent, train_result = train(hp=hp, verbose=True)

    # 2. Save Q-table
    os.makedirs(RESULTS_DIR, exist_ok=True)
    q_table_path = os.path.join(RESULTS_DIR, "q_table.npy")
    agent.save(q_table_path)

    # 3. Evaluate (use a different seed so env varies from training)
    eval_result = evaluate(agent, n_episodes=args.eval_episodes, seed=999, verbose=True)

    # 4. Generate plots
    if not args.no_plots:
        generate_all_plots(
            episode_records=train_result.episode_records,
            q_table=agent.q_table,
            output_dir=PLOTS_DIR,
        )

    # 5. Export full results JSON (for web dashboard)
    export = {
        "meta": {
            "training_time_sec": train_result.training_time_sec,
            "episodes_trained":  train_result.episodes_trained,
            "final_epsilon":     train_result.final_epsilon,
        },
        "hyperparameters": train_result.hyperparameters,
        "evaluation": {
            "n_episodes":   eval_result.n_episodes,
            "avg_reward":   eval_result.avg_reward,
            "std_reward":   eval_result.std_reward,
            "min_reward":   eval_result.min_reward,
            "max_reward":   eval_result.max_reward,
            "avg_steps":    eval_result.avg_steps,
            "success_rate": eval_result.success_rate,
        },
        # Sampled for dashboard (every 10th episode to keep JSON small)
        "training_log": train_result.episode_records[::10],
        "eval_episodes": eval_result.episodes[:20],  # first 20 eval eps
    }

    json_path = os.path.join(RESULTS_DIR, "training_log.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n[OK] Results exported -> {json_path}")

    print("\n" + "=" * 60)
    print("  [DONE] ALL DONE!")
    print(f"  Open web/index.html in your browser to view the dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
