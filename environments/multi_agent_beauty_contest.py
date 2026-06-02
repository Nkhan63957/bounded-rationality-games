"""
Multi-Agent Beauty Contest Environment
=======================================
Architecture: Self-play with parameter sharing.
  - One PPO learner, (n_agents-1) opponents reference the SAME live model.
  - No deepcopy — opponents always use current policy weights.
  - This is true parameter sharing: one network, all agents.

EXP006 comparison target: EXP002 (p=2/3, mixed fixed opponents) -> level 2.726
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import math
import time
import csv
import os


# ── Implied Level Utility ──────────────────────────────────────────────────

def implied_level(mean_sub: float, p: float, level_cap: float = 10.0) -> float:
    if mean_sub <= 0.01:
        return level_cap
    try:
        k = math.log(mean_sub / 50.0) / math.log(p)
        return min(max(k, 0.0), level_cap)
    except (ValueError, ZeroDivisionError):
        return level_cap


# ── Multi-Agent Beauty Contest Environment ─────────────────────────────────

class MultiAgentBeautyContestEnv(gym.Env):
    """
    Gymnasium environment for multi-agent beauty contest (self-play).

    One learner agent interacts with (n_agents-1) opponents.
    Opponents reference the same live PPO model — true parameter sharing.
    No deepcopy: all agents always run the current policy.

    Observation: [last_mean, last_target, round_progress]
    Action:      continuous submission in [0, 100]
    Reward:      proximity = -|submission - target| + base_reward
    """

    metadata = {"render_modes": []}

    def __init__(self, n_agents=5, p=2/3, max_rounds=250, base_reward=100.0):
        super().__init__()
        self.n_agents = n_agents
        self.p = p
        self.max_rounds = max_rounds
        self.base_reward = base_reward
        self.model = None  # set after PPO creation

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._round = 0
        self._last_mean = 50.0
        self._last_target = p * 50.0
        self._last_learner_sub = 50.0

    def _get_obs(self):
        return np.array(
            [self._last_mean, self._last_target,
             self._round / self.max_rounds],
            dtype=np.float32,
        )

    def _opponent_actions(self):
        obs = self._get_obs().reshape(1, -1)
        actions = []
    
        # Use frozen weights if available, otherwise live model
        policy = self.model
        if hasattr(self, 'frozen_state_dict') and self.frozen_state_dict is not None:
            import torch, io, copy
            if not hasattr(self, '_frozen_model') or self._frozen_model is None:
                import torch, io
                self._frozen_model = PPO("MlpPolicy", self)
                self._frozen_model.policy.load_state_dict(self.model.policy.state_dict())
            self.frozen_state_dict.seek(0)
            self._frozen_model.policy.load_state_dict(
                torch.load(self.frozen_state_dict))
            self.frozen_state_dict.seek(0)
            policy = self._frozen_model
    
        for _ in range(self.n_agents - 1):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=False)
                sub = float(np.clip(action[0][0], 0.0, 100.0))
            else:
                sub = float(np.random.uniform(0, 100))
            actions.append(sub)
        return actions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._round = 0
        self._last_mean = 50.0
        self._last_target = self.p * 50.0
        self._last_learner_sub = 50.0
        return self._get_obs(), {}

    def step(self, action):
        learner_sub = float(np.clip(action[0], 0.0, 100.0))
        opp_subs = self._opponent_actions()
        all_subs = [learner_sub] + opp_subs
        mean_sub = float(np.mean(all_subs))
        target = self.p * mean_sub
        reward = float(-abs(learner_sub - target) + self.base_reward)

        self._last_mean = mean_sub
        self._last_target = target
        self._last_learner_sub = learner_sub
        self._round += 1

        terminated = False
        truncated = self._round >= self.max_rounds
        info = {"mean_sub": mean_sub, "target": target,
                "learner_sub": learner_sub, "opponent_subs": opp_subs}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass


# ── Logging Callback ───────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    def __init__(self, env, p, log_interval=2000, csv_path=None, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.p = p
        self.log_interval = log_interval
        self.csv_path = csv_path
        self._last_log = 0

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            infos = self.locals.get("infos", [{}])
            mean_sub = infos[0].get("mean_sub", self.env._last_mean) if infos else self.env._last_mean
            level = implied_level(mean_sub, self.p)
            rewards = self.locals.get("rewards", [0.0])
            mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
            print(
                f"  Step {self.num_timesteps:>7,} | "
                f"Reward: {mean_reward:>8.3f} | "
                f"Mean sub: {mean_sub:>6.2f} | "
                f"Implied level: {level:.3f}"
            )
            if self.csv_path:
                write_header = not os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["step","mean_reward","mean_sub","implied_level"])
                    if write_header:
                        writer.writeheader()
                    writer.writerow({
                        "step": self.num_timesteps,
                        "mean_reward": mean_reward,
                        "mean_sub": mean_sub,
                        "implied_level": level,
                    })
        return True


# ── Training Entry Point ───────────────────────────────────────────────────

class FrozenOpponentCallback(BaseCallback):
    """Update opponent weights every N rollouts instead of real-time."""
    def __init__(self, env, sync_every_n_rollouts=5, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.sync_every = sync_every_n_rollouts
        self._rollout_count = 0
    
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self._rollout_count += 1
        if self._rollout_count % self.sync_every == 0:
            import torch, io
            buf = io.BytesIO()
            torch.save(self.model.policy.state_dict(), buf)
            buf.seek(0)
            self.env.frozen_state_dict = buf
            if self.verbose > 0:
                print(f"  [frozen sync] step {self.num_timesteps:,}")
        return True

def run_exp006(
    total_timesteps=200_000,
    p=2/3,
    n_agents=5,
    seed=42,
    log_interval=2000,
    save_dir="experiments/working",
):
    print("=" * 60)
    print("  EXP006: Multi-agent beauty contest — parameter sharing PPO")
    print(f"  p={p:.4f} | n_agents={n_agents} | timesteps={total_timesteps:,}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("docs/screenshots", exist_ok=True)

    timestamp = time.strftime("%Y%m%d")
    csv_path = os.path.join(save_dir, f"EXP006_log_{timestamp}.csv")
    model_path = os.path.join(save_dir, "EXP006_ppo_multi_agent_beauty_contest")

    env = MultiAgentBeautyContestEnv(n_agents=n_agents, p=p)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
    )

    # Wire live model reference — no deepcopy, true parameter sharing
    env.model = model

    callback = TrainingCallback(
        env=env, p=p, log_interval=log_interval, csv_path=csv_path)

    print(f"\n  Training started — logging every {log_interval:,} steps\n")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")

    # Final evaluation — 500 rounds, deterministic
    print("\n  Running final evaluation (500 rounds, deterministic)...")
    obs, _ = env.reset()
    mean_subs, rewards_eval = [], []
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        mean_subs.append(info["mean_sub"])
        rewards_eval.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    final_mean_sub = float(np.mean(mean_subs))
    final_level = implied_level(final_mean_sub, p)
    final_reward = float(np.mean(rewards_eval))

    model.save(model_path)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"  {'Experiment':<14} {'Opponents':<20} {'Mean Sub':>9} {'Impl Level':>11} {'Reward':>10}")
    print(f"  {'-'*66}")
    print(f"  {'EXP006':<14} {'All PPO (shared)':<20} {final_mean_sub:>9.3f} {final_level:>11.3f} {final_reward:>10.3f}")
    print(f"  {'EXP002':<14} {'Mixed fixed':<20} {'16.610':>9} {'2.726':>11} {'89.143':>10}")
    print(f"  {'Nagel (1995)':<14} {'Human':<20} {'~24.9':>9} {'~1.8':>11} {'—':>10}")
    print("=" * 60)
    print(f"\n  Final mean submission : {final_mean_sub:.3f}")
    print(f"  Final implied level   : {final_level:.3f}")
    print(f"  Model saved           : {model_path}.zip")
    print(f"  Log saved             : {csv_path}")

    return {
        "final_mean_sub": final_mean_sub,
        "final_implied_level": final_level,
        "final_reward": final_reward,
    }

def run_exp007(
    total_timesteps=200_000,
    p=2/3,
    n_agents=5,
    seed=42,
    log_interval=2000,
    save_dir="experiments/working",
    sync_every=5,
):
    print("=" * 60)
    print("  EXP007: Multi-agent — frozen opponents (sync every N rollouts)")
    print(f"  p={p:.4f} | n_agents={n_agents} | sync_every={sync_every}")
    print(f"  Hypothesis: frozen opponents break Nash, produce bounded rationality")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d")
    csv_path = os.path.join(save_dir, f"EXP007_log_{timestamp}.csv")
    model_path = os.path.join(save_dir, "EXP007_ppo_frozen_opponents")

    env = MultiAgentBeautyContestEnv(n_agents=n_agents, p=p)

    model = PPO(
        policy="MlpPolicy", env=env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, verbose=0, seed=seed,
    )

    env.model = model
    env.frozen_state_dict = None

    train_callback = TrainingCallback(
        env=env, p=p, log_interval=log_interval, csv_path=csv_path)
    frozen_callback = FrozenOpponentCallback(
        env=env, sync_every_n_rollouts=sync_every)

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([train_callback, frozen_callback])

    print(f"\n  Training started\n")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")

    # Final evaluation
    obs, _ = env.reset()
    mean_subs, rewards_eval = [], []
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        mean_subs.append(info["mean_sub"])
        rewards_eval.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    final_mean_sub = float(np.mean(mean_subs))
    final_level = implied_level(final_mean_sub, p)
    final_reward = float(np.mean(rewards_eval))

    model.save(model_path)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"  {'Experiment':<16} {'Opponents':<22} {'Mean Sub':>9} {'Impl Level':>11}")
    print(f"  {'-'*60}")
    print(f"  {'EXP007':<16} {'Frozen (sync=5)':<22} {final_mean_sub:>9.3f} {final_level:>11.3f}")
    print(f"  {'EXP006':<16} {'Live shared':<22} {'0.284':>9} {'10.000':>11}")
    print(f"  {'EXP002':<16} {'Fixed mixed':<22} {'16.610':>9} {'2.726':>11}")
    print(f"  {'Nagel (1995)':<16} {'Human':<22} {'~24.9':>9} {'~1.8':>11}")
    print("=" * 60)
    print(f"\n  Final mean submission : {final_mean_sub:.3f}")
    print(f"  Final implied level   : {final_level:.3f}")

    return {
        "final_mean_sub": final_mean_sub,
        "final_implied_level": final_level,
        "final_reward": final_reward,
    }


if __name__ == "__main__":
    run_exp007(
        total_timesteps=200_000,
        p=2/3, n_agents=5, seed=42,
        log_interval=2000,
        save_dir="experiments/working",
        sync_every=5,
    )