"""
Beauty Contest PPO Training Script
====================================
Experiment 001: Winner-take-all reward (expected failure)
Experiment 002: Proximity reward (expected to work)


Each experiment logs to experiments.csv and saves training
curves to docs/screenshots/ for the Maker Portfolio.

Run experiment 001 first, document the failure, then 002.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from environments.p_beauty_contest import PBeautyContest
import os
import csv
from datetime import datetime

# ── Experiment config ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "id":            "EXP001",
        "reward_type":   "winner_take_all",
        "opponent_type": "mixed",
        "n_players":     5,
        "p":             2/3,
        "lr":            3e-4,
        "timesteps":     50_000,
        "note":          "Baseline failure condition — sparse reward expected",
    },
    {
        "id":            "EXP002",
        "reward_type":   "proximity",
        "opponent_type": "mixed",
        "n_players":     5,
        "p":             2/3,
        "lr":            3e-4,
        "timesteps":     200_000,
        "note":          "Dense reward — primary training condition",
    },
    {
        "id":            "EXP003",
        "reward_type":   "proximity",
        "opponent_type": "level1",
        "n_players":     5,
        "p":             2/3,
        "lr":            3e-4,
        "timesteps":     200_000,
        "note":          "Fixed Level-1 opponents — does gap to Nagel close?",
    },
    {
        "id":            "EXP004",
        "reward_type":   "proximity",
        "opponent_type": "mixed",
        "n_players":     5,
        "p":             0.5,
        "lr":            3e-4,
        "timesteps":     200_000,
        "note":          "p=0.5 — does reasoning depth increase as theory predicts?",
    },
    {
        "id":            "EXP005",
        "reward_type":   "proximity",
        "opponent_type": "mixed",
        "n_players":     5,
        "p":             0.9,
        "lr":            3e-4,
        "timesteps":     200_000,
        "note":          "p=0.9 — does reasoning depth decrease as theory predicts?",
    },
    {
        "id":            "EXP005b",
        "reward_type":   "proximity",
        "opponent_type": "mixed",
        "n_players":     5,
        "p":             0.9,
        "lr":            3e-4,
        "timesteps":     500_000,
        "note":          "p=0.9 extended — EXP005 had not converged at 200k",
    },
]

# ── Callback to track training ────────────────────────────────────────────────
class TrainingTracker(BaseCallback):
    """
    Records mean reward and mean submission every eval_freq steps.
    Used to plot training curves and detect convergence or collapse.
    """
    def __init__(self, eval_env, eval_freq=2000, verbose=0):
        super().__init__(verbose)
        self.eval_env   = eval_env
        self.eval_freq  = eval_freq
        self.rewards    = []
        self.steps      = []
        self.mean_subs  = []
        self.impl_levels= []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            total_r = 0.0
            subs, levels = [], []
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, _, info = self.eval_env.step(action)
                total_r += r
                subs.append(info["own_submission"])
                levels.append(info["implied_level"])

            self.rewards.append(total_r)
            self.steps.append(self.n_calls)
            self.mean_subs.append(np.mean(subs))
            self.impl_levels.append(np.mean(levels))

            if self.verbose:
                print(f"  Step {self.n_calls:>7,} | "
                      f"Reward: {total_r:>7.3f} | "
                      f"Mean sub: {np.mean(subs):>6.2f} | "
                      f"Implied level: {np.mean(levels):>5.3f}")
        return True


# ── Training function ─────────────────────────────────────────────────────────
def run_experiment(cfg: dict):
    print(f"\n{'='*60}")
    print(f"  {cfg['id']}: {cfg['note']}")
    print(f"  reward_type={cfg['reward_type']} | "
          f"timesteps={cfg['timesteps']:,} | lr={cfg['lr']}")
    print(f"{'='*60}")

    # Training environment
    train_env = PBeautyContest(
        n_players    = cfg["n_players"],
        p            = cfg["p"],
        reward_type  = cfg["reward_type"],
        opponent_type= cfg["opponent_type"],
        max_rounds   = 100,
        seed         = 42,
    )

    # Evaluation environment (separate instance, deterministic eval)
    eval_env = PBeautyContest(
        n_players    = cfg["n_players"],
        p            = cfg["p"],
        reward_type  = cfg["reward_type"],
        opponent_type= cfg["opponent_type"],
        max_rounds   = 100,
        seed         = 123,
    )

    tracker = TrainingTracker(eval_env, eval_freq=2000, verbose=1)

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate = cfg["lr"],
        n_steps       = 256,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        verbose       = 0,
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        seed          = 42,
    )

    model.learn(total_timesteps=cfg["timesteps"],
                callback=tracker,
                progress_bar=True)

    # ── Final evaluation ──────────────────────────────────────────────────────
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = eval_env.step(action)

    df = eval_env.get_history_df()
    final_mean_sub   = float(df["own_submission"].mean())
    final_impl_level = float(df["implied_level"].mean())
    final_reward     = float(df["reward"].sum())

    print(f"\n  Final mean submission:   {final_mean_sub:.3f}")
    print(f"  Final implied level:     {final_impl_level:.3f}")
    print(f"  Final total reward:      {final_reward:.3f}")
    print(f"  Nagel (1995) human mean: ~24.9 (implied level ~1.8)")

    # ── Save training curves ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"{cfg['id']} — {cfg['reward_type']} reward\n"
                 f"n_players={cfg['n_players']}, p={cfg['p']:.3f}",
                 fontsize=13)

    axes[0].plot(tracker.steps, tracker.rewards, color='#8b5cf6', linewidth=2)
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Curve — Reward over Timesteps")
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tracker.steps, tracker.mean_subs, color='#34d399', linewidth=2)
    axes[1].axhline(y=50.0, color='gray', linestyle='--',
                    alpha=0.5, label='Level-0 (50)')
    axes[1].axhline(y=33.3, color='#f59e0b', linestyle='--',
                    alpha=0.7, label='Level-1 (33.3)')
    axes[1].axhline(y=22.2, color='#f87171', linestyle='--',
                    alpha=0.7, label='Level-2 (22.2)')
    axes[1].axhline(y=24.9, color='#60a5fa', linestyle='-',
                    alpha=0.9, label='Nagel human mean (24.9)')
    axes[1].set_ylabel("Mean Submission")
    axes[1].set_title("Mean Submission over Timesteps")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tracker.steps, tracker.impl_levels,
                 color='#fb923c', linewidth=2)
    axes[2].axhline(y=1.8, color='#60a5fa', linestyle='-',
                    alpha=0.9, label='Nagel human mean (~1.8)')
    axes[2].set_ylabel("Implied Reasoning Level")
    axes[2].set_xlabel("Training Timesteps")
    axes[2].set_title("Implied Reasoning Level over Timesteps")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"docs/screenshots/{datetime.now().strftime('%Y%m%d')}_{cfg['id']}_training_curve.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curve saved → {fname}")

    # ── Log to experiments.csv ────────────────────────────────────────────────
    with open("experiments.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"),
            cfg["id"],
            "p_beauty_contest",
            cfg["n_players"],
            cfg["p"],
            "PPO",
            cfg["lr"],
            cfg["timesteps"],
            cfg["reward_type"],
            round(final_mean_sub, 4),
            round(final_impl_level, 4),
            cfg["note"],
            "COMPLETE",
        ])

    # ── Save model ────────────────────────────────────────────────────────────
    save_path = f"experiments/working/{cfg['id']}_ppo_beauty_contest"
    model.save(save_path)
    print(f"  Model saved → {save_path}.zip")

    return {
        "id":           cfg["id"],
        "reward_type":  cfg["reward_type"],
        "final_sub":    final_mean_sub,
        "final_level":  final_impl_level,
        "final_reward": final_reward,
        "tracker":      tracker,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    # Run EXP001 first — expected failure
    # Winner-take-all reward is too sparse for early learning.
    # Agent wins <5% of rounds during initial exploration.
    # Expected: flat training curve, no convergence.
    # results["EXP001"] = run_experiment(EXPERIMENTS[0])

    # print(f"\n{'='*60}")
    # print("  EXP001 complete. Check training curve before proceeding.")
    # print(f"  Expected: flat reward curve (sparse signal failure)")
    # print(f"  If curve is flat → proceed to EXP002 (proximity reward)")
    # print(f"{'='*60}\n")

    # Run EXP002 — proximity reward
    # results["EXP002"] = run_experiment(EXPERIMENTS[1])

    # Run EXP003 — fixed level-1 opponents
    # results["EXP003"] = run_experiment(EXPERIMENTS[2])
    # results["EXP004"] = run_experiment(EXPERIMENTS[3])
    # results["EXP005"] = run_experiment(EXPERIMENTS[4])
    results["EXP005b"] = run_experiment(EXPERIMENTS[5])

    # ── Comparison summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Experiment':<12} {'Reward Type':<20} "
          f"{'Mean Sub':>10} {'Impl Level':>12} {'Total Reward':>13}")
    print(f"  {'-'*70}")
    for k, r in results.items():
        print(f"  {r['id']:<12} {r['reward_type']:<20} "
              f"{r['final_sub']:>10.3f} {r['final_level']:>12.3f} "
              f"{r['final_reward']:>13.3f}")
    print(f"\n  Nagel (1995) human benchmark:")
    print(f"  {'Human subjects':<12} {'—':<20} "
          f"{'~24.9':>10} {'~1.8':>12} {'—':>13}")
    print(f"{'='*60}")