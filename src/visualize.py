"""
Visualization module: learning curves, Q-value heatmaps, epsilon decay.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Custom dark-mode color palette ─────────────────────────────────────
BG_COLOR    = "#0d1117"
SURFACE     = "#161b22"
ACCENT1     = "#58a6ff"    # blue
ACCENT2     = "#3fb950"    # green
ACCENT3     = "#f78166"    # red/orange
ACCENT4     = "#d2a8ff"    # purple
TEXT_COLOR  = "#e6edf3"
GRID_COLOR  = "#21262d"

def _apply_dark_style(fig, axes_list=None):
    """Apply consistent dark-mode styling to a figure."""
    fig.patch.set_facecolor(BG_COLOR)
    if axes_list is None:
        axes_list = fig.get_axes()
    for ax in axes_list:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.8)


def _rolling_avg(data, window: int = 50):
    """Compute rolling average with valid convolution."""
    if len(data) < window:
        return np.cumsum(data) / (np.arange(len(data)) + 1)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_learning_curve(
    episode_records: list,
    window: int = 100,
    save_path: str = None,
) -> str:
    """
    Plot reward per episode with a rolling average overlay.

    Returns:
        Path to the saved PNG (or empty string if not saved).
    """
    rewards  = [r["total_reward"] for r in episode_records]
    episodes = [r["episode"]      for r in episode_records]

    rolling = _rolling_avg(rewards, window)
    roll_x  = episodes[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_style(fig)

    # Raw rewards (faint)
    ax.plot(episodes, rewards, color=ACCENT1, alpha=0.15, linewidth=0.8, label="Episode Reward")
    # Rolling average
    ax.plot(roll_x, rolling, color=ACCENT2, linewidth=2.2,
            label=f"Rolling Avg (w={window})")

    ax.axhline(y=0,  color=GRID_COLOR, linewidth=1.0, linestyle="--")
    ax.axhline(y=8,  color=ACCENT3, linewidth=1.0, linestyle=":",
               alpha=0.6, label="Convergence target (+8)")

    ax.set_title("🚕  Q-Learning: Reward per Episode", fontsize=15, pad=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    legend = ax.legend(facecolor=SURFACE, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.set_xlim(1, episodes[-1])

    plt.tight_layout()
    out = ""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        out = save_path
    plt.close(fig)
    return out


def plot_steps_curve(
    episode_records: list,
    window: int = 100,
    save_path: str = None,
) -> str:
    """Plot steps-to-completion per episode with rolling average."""
    steps    = [r["steps"]   for r in episode_records]
    episodes = [r["episode"] for r in episode_records]

    rolling = _rolling_avg(steps, window)
    roll_x  = episodes[window - 1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_style(fig)

    ax.plot(episodes, steps, color=ACCENT4, alpha=0.15, linewidth=0.8, label="Steps/Episode")
    ax.plot(roll_x, rolling, color=ACCENT1, linewidth=2.2,
            label=f"Rolling Avg (w={window})")

    ax.set_title("🚕  Q-Learning: Steps to Completion per Episode", fontsize=15, pad=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps", fontsize=12)
    ax.legend(facecolor=SURFACE, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.set_xlim(1, episodes[-1])

    plt.tight_layout()
    out = ""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        out = save_path
    plt.close(fig)
    return out


def plot_epsilon_decay(
    episode_records: list,
    save_path: str = None,
) -> str:
    """Plot exploration rate (epsilon) over episodes."""
    eps      = [r["epsilon"] for r in episode_records]
    episodes = [r["episode"] for r in episode_records]

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_style(fig)

    ax.plot(episodes, eps, color=ACCENT3, linewidth=2.0, label="Epsilon (ε)")
    ax.fill_between(episodes, eps, alpha=0.15, color=ACCENT3)

    ax.set_title("📉  Epsilon Decay (Exploration Rate)", fontsize=15, pad=14)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=SURFACE, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.set_xlim(1, episodes[-1])

    plt.tight_layout()
    out = ""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        out = save_path
    plt.close(fig)
    return out


def plot_q_heatmap(
    q_table: np.ndarray,
    save_path: str = None,
) -> str:
    """
    Visualize the Q-table as a heatmap: max Q-value per state.
    States are arranged in the natural Taxi-v3 order (500 states).
    """
    max_q = np.max(q_table, axis=1)   # (500,) best value per state

    # Reshape into a 2D grid for visual clarity: 25 cols × 20 rows
    rows, cols = 20, 25
    grid = max_q[:rows * cols].reshape(rows, cols)

    cmap = LinearSegmentedColormap.from_list(
        "taxi", ["#0d1117", "#1f6feb", "#58a6ff", "#3fb950"]
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_dark_style(fig)

    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    cbar.set_label("Max Q-Value", color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_title("🗺  Q-Table Heatmap: Best Q-Value per State", fontsize=15, pad=14)
    ax.set_xlabel("State (mod 25)", fontsize=12)
    ax.set_ylabel("State Group", fontsize=12)

    plt.tight_layout()
    out = ""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        out = save_path
    plt.close(fig)
    return out


def plot_action_distribution(
    q_table: np.ndarray,
    save_path: str = None,
) -> str:
    """Bar chart: how often each action is the greedy best action."""
    ACTION_NAMES = ["⬇ S", "⬆ N", "➡ E", "⬅ W", "🚕 Pick", "📦 Drop"]
    best_actions = np.argmax(q_table, axis=1)
    counts = np.bincount(best_actions, minlength=6)
    pcts   = counts / counts.sum() * 100

    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#ffa657", "#ff7b72"]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig)

    bars = ax.bar(ACTION_NAMES, pcts, color=colors, width=0.6, edgecolor=BG_COLOR)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center", va="bottom", color=TEXT_COLOR, fontsize=10,
        )

    ax.set_title("🎯  Greedy Action Distribution Across All States", fontsize=15, pad=14)
    ax.set_xlabel("Action", fontsize=12)
    ax.set_ylabel("% of States", fontsize=12)
    ax.set_ylim(0, pcts.max() * 1.2)

    plt.tight_layout()
    out = ""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        out = save_path
    plt.close(fig)
    return out


def generate_all_plots(episode_records: list, q_table: np.ndarray, output_dir: str):
    """Generate and save all plots to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    paths["learning_curve"]      = plot_learning_curve(
        episode_records, save_path=os.path.join(output_dir, "learning_curve.png"))
    paths["steps_curve"]         = plot_steps_curve(
        episode_records, save_path=os.path.join(output_dir, "steps_curve.png"))
    paths["epsilon_decay"]       = plot_epsilon_decay(
        episode_records, save_path=os.path.join(output_dir, "epsilon_decay.png"))
    paths["q_heatmap"]           = plot_q_heatmap(
        q_table, save_path=os.path.join(output_dir, "q_heatmap.png"))
    paths["action_distribution"] = plot_action_distribution(
        q_table, save_path=os.path.join(output_dir, "action_distribution.png"))

    print(f"\n📈  All plots saved to: {output_dir}")
    for name, path in paths.items():
        print(f"   • {name}: {path}")

    return paths
