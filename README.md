# 🚕 Taxi Route Optimization with Reinforcement Learning

A complete **Q-Learning** agent that solves the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment — learning optimal pickup and drop-off routes from scratch.

---

Live website link: https://taxi-route-optimization-with.onrender.com/

## 🧠 Algorithm Overview

| Component | Details |
|-----------|---------|
| **Method** | Tabular Q-Learning |
| **State Space** | 500 states (5×5 grid × 5 passenger locs × 4 destinations) |
| **Action Space** | 6 actions (N, S, E, W, Pickup, Dropoff) |
| **Policy** | ε-greedy (exponential decay) |
| **Update Rule** | Bellman equation |

**Bellman Q-Update:**
```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

**Reward Structure:**
- `+20` — successful passenger dropoff  
- `−10` — illegal pickup/dropoff  
- `−1`  — each time step (encourages efficiency)

---

## 📁 Project Structure

```
taxi-rl/
├── src/
│   ├── agent.py        # QLearningAgent class
│   ├── environment.py  # Gymnasium Taxi-v3 wrapper
│   ├── train.py        # Training loop & hyperparameters
│   ├── evaluate.py     # Policy evaluation & metrics
│   └── visualize.py    # Learning curve & Q-table plots
├── web/
│   ├── index.html      # Interactive dashboard
│   ├── style.css       # Dark-mode styling
│   └── app.js          # Chart.js + state explorer
├── results/            # Auto-generated after training
│   ├── q_table.npy
│   ├── training_log.json
│   └── plots/
│       ├── learning_curve.png
│       ├── steps_curve.png
│       ├── epsilon_decay.png
│       ├── q_heatmap.png
│       └── action_distribution.png
├── train_and_export.py # ← Main entry point
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent
```bash
python train_and_export.py
```

### 3. Customize hyperparameters (CLI)
```bash
python train_and_export.py \
  --episodes 5000 \
  --alpha 0.15 \
  --gamma 0.99 \
  --epsilon-decay 0.9997
```

### 4. Open the dashboard
Open `web/index.html` in your browser. The dashboard auto-loads `results/training_log.json`.

---

## ⚙️ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | `0.1` | Learning rate — controls Q-update step size |
| `--gamma` | `0.99` | Discount factor — weight of future rewards |
| `--epsilon` | `1.0` | Initial exploration rate (fully random) |
| `--epsilon-min` | `0.01` | Floor for exploration |
| `--epsilon-decay` | `0.9995` | Per-episode multiplicative decay |
| `--episodes` | `3000` | Training episodes |
| `--eval-episodes` | `100` | Evaluation episodes (greedy policy) |

---

## 📊 Results (default hyperparameters)

After 3000 training episodes:

| Metric | Value |
|--------|-------|
| Avg Reward (eval) | ≥ +8 |
| Success Rate | ≥ 95% |
| Avg Steps | ≈ 13 |
| Training Time | ~12s |

---

## 🖥 Interactive Dashboard Features

- **Learning curves** — reward & steps per episode with rolling average
- **Epsilon decay** — track exploration rate over time
- **Eval reward distribution** — histogram of test episode rewards
- **Q-Table State Explorer** — decode any of 500 states, view Q-values as bars
- **Taxi Grid Renderer** — visualize taxi, passenger, and destination on the 5×5 map
- **Episode Replay** — step through a test episode with play/pause

---

## 📚 References

- [Gymnasium Taxi-v3 docs](https://gymnasium.farama.org/environments/toy_text/taxi/)
- Watkins & Dayan (1992) — Q-Learning
- Sutton & Barto — Reinforcement Learning: An Introduction
