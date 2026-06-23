"""
repeated_game.py — repeated simultaneous-move tariff game (single-country view)

One country is the learner; the partner is played either by a frozen policy
(another PPO model) or a constant tariff. Both move simultaneously each period
based on the previous period's tariffs, so the stage game is the validated
TradeWarGame and the repeated structure permits (in principle) reciprocity.

Reward is the country's own welfare W_i(t_own, t_partner) in neutral model
units — never a shaped/amplified reward. The analytic Nash and cooperative
tariffs from trade_game.py are the benchmarks the learners are scored against.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from trade_game import TradeWarGame


class RepeatedTariffEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, game: TradeWarGame, partner_model=None,
                 partner_const: float = 0.0, horizon: int = 16):
        super().__init__()
        self.game = game
        self.partner_model = partner_model        # frozen PPO policy, or None
        self.partner_const = partner_const        # used when partner_model is None
        self.H = horizon
        # obs: [own_last_tariff, partner_last_tariff, step_fraction]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.], np.float32),
                                            high=np.array([1., 1., 1.], np.float32))
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.own_last = 0.0
        self.partner_last = 0.0
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.array([self.own_last, self.partner_last,
                         self.t / max(self.H - 1, 1)], dtype=np.float32)

    def _partner_tariff(self):
        # partner observes the mirror state: [partner_last, own_last, step_frac]
        if self.partner_model is None:
            return float(self.partner_const)
        pobs = np.array([self.partner_last, self.own_last,
                         self.t / max(self.H - 1, 1)], dtype=np.float32)
        a, _ = self.partner_model.predict(pobs, deterministic=True)
        return self.game.action_to_tariff(float(a[0]))

    def step(self, action):
        tau_own = self.game.action_to_tariff(float(action[0]))
        tau_partner = self._partner_tariff()                 # simultaneous (both from last state)
        reward = self.game.welfare(tau_own, tau_partner)
        self.own_last, self.partner_last = tau_own, tau_partner
        self.t += 1
        done = self.t >= self.H
        return self._obs(), float(reward), done, False, {
            "tau_own": tau_own, "tau_partner": tau_partner}


class CooperativePlannerEnv(gym.Env):
    """One planner sets BOTH tariffs to maximize joint welfare -> benchmark (free trade)."""
    metadata = {"render_modes": []}

    def __init__(self, game: TradeWarGame, horizon: int = 16):
        super().__init__()
        self.game = game; self.H = horizon
        self.observation_space = spaces.Box(low=0., high=1., shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.t = 0; self.a = self.b = 0.0
        return np.array([self.a, self.b, 0.], np.float32), {}

    def step(self, action):
        ta = self.game.action_to_tariff(float(action[0]))
        tb = self.game.action_to_tariff(float(action[1]))
        reward = self.game.welfare(ta, tb) + self.game.welfare(tb, ta)   # joint
        self.a, self.b = ta, tb; self.t += 1
        done = self.t >= self.H
        return np.array([ta, tb, self.t / max(self.H - 1, 1)], np.float32), \
            float(reward), done, False, {"tau_a": ta, "tau_b": tb}
