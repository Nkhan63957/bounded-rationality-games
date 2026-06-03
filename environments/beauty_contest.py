"""
P-Beauty Contest Environment
==============================
A p-beauty contest game where n players simultaneously submit
a number in [0, 100]. The winner is the player whose submission
is closest to p * mean(all submissions).

Classic result (Nagel 1995): human subjects cluster around
33 and 22, corresponding to 1-2 levels of reasoning above
naive play (Level-0: random ~50, Level-1: 33, Level-2: 22).
Nash equilibrium: 0 (everyone reasons to infinity).

This environment is the first test case for whether PPO agents
exhibit bounded rationality consistent with human subjects.

Parameters
----------
n_players : int
    Number of players (default 5, matching standard poker table
    and Nagel's common experimental setup)
p : float
    Multiplier applied to mean submission (default 2/3)
reward_type : str
    'winner_take_all' or 'proximity'
    winner_take_all: +1 to winner, 0 to all others
    proximity: reward inversely proportional to distance
               from winning number (dense gradient signal)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PBeautyContest(gym.Env):
    """
    Single-agent interface to the p-beauty contest.
    The learning agent plays against n-1 fixed opponents
    whose strategy type is specified at initialization.

    Opponent types
    --------------
    'random'   : Level-0, uniform draw from [0, 100]
    'level1'   : Best response to random (= p * 50)
    'level2'   : Best response to level1
    'levelk'   : Best response to level k-1 (computed on init)
    'nash'     : Always submit 0
    'mixed'    : Population of mixed levels (most realistic)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 n_players: int = 5,
                 p: float = 2/3,
                 reward_type: str = "winner_take_all",
                 opponent_type: str = "mixed",
                 max_rounds: int = 100,
                 seed: int = 42):

        super().__init__()

        self.n_players    = n_players
        self.p            = p
        self.reward_type  = reward_type
        self.opponent_type = opponent_type
        self.max_rounds   = max_rounds
        self.rng          = np.random.default_rng(seed)

        # Observation: own last submission + last round mean + round progress
        # Shape: [own_last_submission, last_mean, last_target, round_progress]
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0.], dtype=np.float32),
            high=np.array([100., 100., 100., 1.], dtype=np.float32),
            dtype=np.float32
        )

        # Action: submit a number in [0, 100]
        self.action_space = spaces.Box(
            low=np.array([0.], dtype=np.float32),
            high=np.array([100.], dtype=np.float32),
            dtype=np.float32
        )

        # Precompute level-k submissions
        self._level_k = self._compute_level_k(max_k=10)

        self._round      = 0
        self._last_mean  = 50.0
        self._last_target = self.p * 50.0
        self._own_last   = 50.0
        self._history    = []

    def _compute_level_k(self, max_k: int = 10) -> dict:
        """
        Compute deterministic level-k submissions.
        Level-0: 50 (midpoint of uniform distribution)
        Level-k: p * Level-(k-1)
        """
        levels = {0: 50.0}
        for k in range(1, max_k + 1):
            levels[k] = self.p * levels[k - 1]
        return levels

    def _get_opponent_submissions(self) -> np.ndarray:
        """
        Generate n-1 opponent submissions based on opponent_type.
        """
        n_opp = self.n_players - 1

        if self.opponent_type == "random":
            return self.rng.uniform(0, 100, n_opp).astype(np.float32)

        elif self.opponent_type == "level1":
            return np.full(n_opp, self._level_k[1], dtype=np.float32)

        elif self.opponent_type == "level2":
            return np.full(n_opp, self._level_k[2], dtype=np.float32)

        elif self.opponent_type == "nash":
            return np.zeros(n_opp, dtype=np.float32)

        elif self.opponent_type == "mixed":
            # Realistic population: 40% L1, 35% L2, 15% L0, 10% L3+
            # Based on Nagel (1995) empirical distribution
            submissions = []
            for _ in range(n_opp):
                r = self.rng.random()
                if r < 0.40:
                    submissions.append(self._level_k[1])
                elif r < 0.75:
                    submissions.append(self._level_k[2])
                elif r < 0.90:
                    submissions.append(self.rng.uniform(0, 100))
                else:
                    submissions.append(self._level_k[3])
            return np.array(submissions, dtype=np.float32)

        else:
            raise ValueError(f"Unknown opponent_type: {self.opponent_type}")

    def _compute_reward(self,
                        own_submission: float,
                        all_submissions: np.ndarray,
                        target: float) -> float:
        """
        Compute reward for the learning agent.

        winner_take_all: +1 if closest to target, 0 otherwise.
                         Sparse — provides almost no gradient
                         signal during early training. Kept as
                         the baseline failure condition.

        proximity:       reward = 1 - (distance / 100)
                         Dense signal — agent always receives
                         feedback proportional to performance.
        """
        distances = np.abs(all_submissions - target)
        own_dist  = abs(own_submission - target)

        if self.reward_type == "winner_take_all":
            return 1.0 if own_dist == distances.min() else 0.0

        elif self.reward_type == "proximity":
            return float(1.0 - own_dist / 100.0)

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self._own_last,
            self._last_mean,
            self._last_target,
            self._round / self.max_rounds,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._round      = 0
        self._last_mean  = 50.0
        self._last_target = self.p * 50.0
        self._own_last   = 50.0
        self._history    = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        own_submission = float(np.clip(action[0], 0., 100.))
        opp_submissions = self._get_opponent_submissions()
        all_submissions = np.append(opp_submissions, own_submission)

        mean_submission = float(all_submissions.mean())
        target          = self.p * mean_submission
        reward          = self._compute_reward(
                              own_submission, all_submissions, target)

        self._last_mean   = mean_submission
        self._last_target = target
        self._own_last    = own_submission
        self._round      += 1

        self._history.append({
            "round":          self._round,
            "own_submission": own_submission,
            "mean":           mean_submission,
            "target":         target,
            "reward":         reward,
            "implied_level":  self._implied_level(own_submission),
        })

        terminated = self._round >= self.max_rounds
        return self._get_obs(), reward, terminated, False, {
            "own_submission": own_submission,
            "mean":           mean_submission,
            "target":         target,
            "implied_level":  self._implied_level(own_submission),
        }

    def _implied_level(self, submission: float) -> float:
        """
        Infer implied reasoning level from submission.
        Level-k submission = p^k * 50, so:
        k = log(submission / 50) / log(p)
        Clipped to [0, 10] for numerical stability.
        """
        if submission <= 0:
            return 10.0  # Nash play
        if submission >= 100:
            return 0.0   # Above Level-0
        try:
            level = np.log(submission / 50.) / np.log(self.p)
            return float(np.clip(level, 0., 10.))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def get_history_df(self):
        """Return episode history as a DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame(self._history)