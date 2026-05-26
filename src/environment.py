"""
Gymnasium Taxi-v3 environment wrapper with state decoding utilities.
"""
import gymnasium as gym
import numpy as np


# State encoding: state = row * 100 + col * 20 + passenger * 4 + destination
TAXI_ROW_STATES   = 5
TAXI_COL_STATES   = 5
PASSENGER_LOCS    = 5  # 0-3 = grid locations, 4 = in taxi
DEST_LOCS         = 4

PASSENGER_LABELS  = ["R", "G", "Y", "B", "Taxi"]
DESTINATION_LABELS = ["R", "G", "Y", "B"]

# Fixed map positions on the 5x5 grid
SPECIAL_POSITIONS = {
    "R": (0, 0),
    "G": (0, 4),
    "Y": (4, 0),
    "B": (4, 3),
}


class TaxiEnvironment:
    """
    Thin wrapper around gymnasium's Taxi-v3 environment.

    Provides a clean API for training and rendering,
    plus human-readable state decoding utilities.
    """

    def __init__(self, render_mode: str = None, seed: int = 42):
        """
        Args:
            render_mode: None (headless), 'ansi' (text), or 'rgb_array'.
            seed:        RNG seed for the *first* reset (subsequent resets are random).
        """
        self.render_mode = render_mode
        self.seed = seed
        self._first_reset = True
        self.env = gym.make("Taxi-v3", render_mode=render_mode)

        self.n_states  = self.env.observation_space.n   # 500
        self.n_actions = self.env.action_space.n        # 6

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Reset environment, return initial state.
        Uses the configured seed only on the first reset for reproducibility,
        then lets the environment's internal RNG vary subsequent episodes.
        """
        if self._first_reset:
            state, _ = self.env.reset(seed=self.seed)
            self._first_reset = False
        else:
            state, _ = self.env.reset()
        return int(state)

    def step(self, action: int):
        """
        Take a step.

        Returns:
            (next_state, reward, done, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return int(next_state), float(reward), bool(done), info

    def render(self) -> str:
        """Return rendered frame (only useful when render_mode='ansi')."""
        return self.env.render()

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # State decoding
    # ------------------------------------------------------------------

    @staticmethod
    def decode_state(state: int) -> dict:
        """
        Decode an integer state into human-readable components.

        The Taxi-v3 encoding (from gym source):
            state = ((row * 5 + col) * 5 + passenger_loc) * 4 + destination

        Returns a dict with keys: row, col, passenger, destination,
            passenger_label, destination_label.
        """
        dest    = state % 4;          state //= 4
        pass_   = state % 5;          state //= 5
        col     = state % 5;          state //= 5
        row     = state

        return {
            "row":               row,
            "col":               col,
            "passenger_idx":     pass_,
            "destination_idx":   dest,
            "passenger_label":   PASSENGER_LABELS[pass_],
            "destination_label": DESTINATION_LABELS[dest],
        }

    @staticmethod
    def decode_state_str(state: int) -> str:
        """Return a compact human-readable string for a state."""
        d = TaxiEnvironment.decode_state(state)
        return (
            f"Taxi({d['row']},{d['col']}) "
            f"Passenger={d['passenger_label']} "
            f"Dest={d['destination_label']}"
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"TaxiEnvironment("
            f"n_states={self.n_states}, n_actions={self.n_actions}, "
            f"render_mode={self.render_mode!r})"
        )
