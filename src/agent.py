"""
Q-Learning Agent for Taxi-v3
Implements epsilon-greedy exploration and Bellman equation updates.
"""
import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    State space : 500  (5x5 grid × 5 passenger locations × 4 destinations)
    Action space:   6  (South, North, East, West, Pickup, Dropoff)

    Q-update rule (Bellman equation):
        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
    """

    ACTION_NAMES = {
        0: "⬇ South",
        1: "⬆ North",
        2: "➡ East",
        3: "⬅ West",
        4: "🚕 Pickup",
        5: "📦 Dropoff",
    }

    def __init__(
        self,
        n_states: int = 500,
        n_actions: int = 6,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        random_seed: int = 42,
    ):
        """
        Args:
            n_states:       Number of discrete states in the environment.
            n_actions:      Number of discrete actions.
            alpha:          Learning rate (0, 1].
            gamma:          Discount factor [0, 1].
            epsilon:        Initial exploration rate.
            epsilon_min:    Minimum exploration rate.
            epsilon_decay:  Multiplicative decay applied after each episode.
            random_seed:    For reproducibility.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(random_seed)

        # Q-table: shape (n_states, n_actions), initialized to zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        # Training counters
        self.episodes_trained = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state:  Current environment state.
            greedy: If True, always exploit (used during evaluation).

        Returns:
            Selected action index.
        """
        if not greedy and self.rng.random() < self.epsilon:
            # Explore: random action
            return int(self.rng.integers(0, self.n_actions))
        else:
            # Exploit: action with highest Q-value (break ties randomly)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return int(self.rng.choice(best_actions))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Apply the Bellman Q-update.

        Returns:
            TD error (temporal difference error) for this step.
        """
        # Target: r + γ · max Q(s', a')  (0 if terminal)
        future_value = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * future_value

        # Current estimate
        td_error = td_target - self.q_table[state, action]

        # Update
        self.q_table[state, action] += self.alpha * td_error
        self.total_steps += 1

        return float(td_error)

    def decay_epsilon(self):
        """Apply exponential epsilon decay (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def best_action(self, state: int) -> int:
        """Return the greedy best action for a given state."""
        return int(np.argmax(self.q_table[state]))

    def best_action_name(self, state: int) -> str:
        """Return human-readable best action name for a given state."""
        return self.ACTION_NAMES[self.best_action(state)]

    def q_values_for_state(self, state: int) -> dict:
        """Return dict mapping action names to Q-values."""
        return {
            self.ACTION_NAMES[a]: round(float(self.q_table[state, a]), 4)
            for a in range(self.n_actions)
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table to a .npy file."""
        np.save(path, self.q_table)
        print(f"✅ Q-table saved → {path}")

    def load(self, path: str):
        """Load Q-table from a .npy file."""
        self.q_table = np.load(path)
        print(f"✅ Q-table loaded ← {path}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"QLearningAgent("
            f"α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, episodes={self.episodes_trained})"
        )
