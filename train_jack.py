"""
Jack of Hearts — PPO Training Script
======================================
EXP-JH001: Jack PPO agent vs rule-based regular players.

Architecture:
  - Jack: PPO with custom 256x256 MlpPolicy
  - Regular players: RuleBasedRegular agents
  - 500,000 timesteps
  - Log every 5,000 steps

Metrics tracked:
  - Lobby wipe rate (Jack wins)
  - Jack cornered rate (Regular wins)
  - Jack voluntary concession rate
  - Average game duration (rounds)
  - Alliance formation rate
  - Mean reward per episode

Research question:
  What strategy does the Jack agent discover against
  rule-based regular players? Does it converge to:
  (a) Active deception — manipulate and deflect
  (b) Passive honesty — blend in, let regulars die
  (c) Mixed strategy — shift as population shrinks
"""

import numpy as np
import os
import csv
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.jack_of_hearts import JackOfHeartsEnv, N_PLAYERS
from environments.game_coordinator import GameCoordinator
from agents.rule_based_regular import RuleBasedRegular


# ── Single-environment wrapper for SB3 ────────────────────────────────────

class JackTrainingEnv(JackOfHeartsEnv):
    """
    Wraps JackOfHeartsEnv for single-agent PPO training.
    Jack agent trains against rule-based regular players.
    Full game logic handled internally per episode.
    """

    def __init__(self, seed: int = 42):
        super().__init__(agent_role="jack", agent_id=0,
                         n_perceptive=2, seed=seed)
        self.coord = None
        self.rule_based = {}
        self._episode_rounds = 0
        self._episode_reward = 0.0
        self._last_info = {}

    def reset(self, seed=None, options=None):
        # Build fresh coordinator each episode
        self.coord = GameCoordinator(n_perceptive=2,
                                     seed=np.random.randint(0, 10000))
        obs_dict = self.coord.reset()

        # Build rule-based agents for this episode
        self.rule_based = {}
        for i in range(1, N_PLAYERS):
            self.rule_based[i] = RuleBasedRegular(
                player_id=i,
                neighborhood=self.coord.envs[0].neighborhoods[i],
                seed=42 + i,
            )

        self._episode_rounds = 0
        self._episode_reward = 0.0
        self._last_info = {}

        # Sync own state with coordinator
        self._sync_from_coord()

        return obs_dict[0], {}

    def step(self, action: np.ndarray):
        """
        One PPO step = one full game round.
        Jack uses PPO action, regulars use rule-based.
        """
        if self.coord.done:
            obs, _ = self.reset()
            return obs, 0.0, True, False, self._last_info

        # Collect all actions
        actions = {0: action}

        obs_dict = self._get_obs_dict()
        env_state = {
            "alive_mask":    self.coord.envs[0].alive_mask,
            "symbols":       self.coord.envs[0].symbols,
            "player_states": self.coord.envs[0].players,
            "round":         self.coord.round,
        }

        for i in self.coord.alive_players:
            if i != 0:
                actions[i] = self.rule_based[i].act(
                    obs_dict[i], env_state)

        # Step coordinator
        obs_dict, rewards, done, info = self.coord.step(actions)

        jack_reward = rewards.get(0, 0.0)
        self._episode_rounds += 1
        self._episode_reward += jack_reward
        self._last_info = info

        # Sync own state
        self._sync_from_coord()

        jack_obs = obs_dict.get(0, self._get_obs())

        # Augment info with episode metrics
        info["episode_rounds"]  = self._episode_rounds
        info["episode_reward"]  = self._episode_reward
        info["n_alive"]         = len(self.coord.alive_players)

        if done:
            info["terminal_observation"] = jack_obs

        return jack_obs, jack_reward, done, False, info

    def _sync_from_coord(self):
        """Sync env state from coordinator."""
        src = self.coord.envs[0]
        self.players       = src.players
        self.symbols       = src.symbols
        self.alive_mask    = src.alive_mask
        self.round         = src.round
        self.game_over     = src.game_over
        self.winner        = src.winner
        self.last_messages = src.last_messages
        self.neighborhoods = src.neighborhoods

    def _get_obs_dict(self):
        """Get observations for all alive players."""
        obs = {}
        src = self.coord.envs[0]
        for i in self.coord.alive_players:
            self.coord.envs[i].round         = src.round
            self.coord.envs[i].alive_mask    = src.alive_mask
            self.coord.envs[i].last_messages = src.last_messages
            self.coord.envs[i].players       = src.players
            obs[i] = self.coord.envs[i]._get_obs()
        return obs


# ── Training Callback ──────────────────────────────────────────────────────

class JackTrainingCallback(BaseCallback):
    """
    Logs training metrics every log_interval timesteps.
    Tracks: lobby wipe rate, cornered rate, concession rate,
    average game duration, alliance formation, reward.
    """

    def __init__(self, log_interval: int = 5000,
                 csv_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.csv_path     = csv_path
        self._last_log    = 0

        # Episode tracking
        self._episode_outcomes = []
        self._episode_durations = []
        self._episode_alliances = []
        self._episode_rewards   = []

    def _on_step(self) -> bool:
        # Collect episode info from locals
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info.get("outcome") in (
                "jack_wins_lobby_wipe",
                "regular_wins_suspicion_consensus",
                "jack_voluntary_concession",
            ):
                self._episode_outcomes.append(info.get("outcome"))
                self._episode_durations.append(
                    info.get("episode_rounds", 0))
                self._episode_alliances.append(
                    info.get("n_alliances", 0))
                self._episode_rewards.append(
                    info.get("episode_reward", 0.0))

        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            self._log_metrics()

        return True

    def _log_metrics(self):
        n = len(self._episode_outcomes)
        if n == 0:
            print(f"  Step {self.num_timesteps:>8,} | "
                  f"No completed episodes yet")
            return

        wipe_rate    = self._episode_outcomes.count(
            "jack_wins_lobby_wipe") / n
        corner_rate  = self._episode_outcomes.count(
            "regular_wins_suspicion_consensus") / n
        concede_rate = self._episode_outcomes.count(
            "jack_voluntary_concession") / n
        avg_duration = np.mean(self._episode_durations)
        avg_alliances = np.mean(self._episode_alliances)
        avg_reward   = np.mean(self._episode_rewards)

        print(
            f"  Step {self.num_timesteps:>8,} | "
            f"Episodes: {n:>4} | "
            f"Wipe: {wipe_rate:.2f} | "
            f"Cornered: {corner_rate:.2f} | "
            f"Concede: {concede_rate:.2f} | "
            f"Avg rounds: {avg_duration:.1f} | "
            f"Reward: {avg_reward:.3f}"
        )

        if self.csv_path:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                fieldnames = [
                    "step", "n_episodes", "wipe_rate",
                    "corner_rate", "concede_rate",
                    "avg_duration", "avg_alliances", "avg_reward"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "step":          self.num_timesteps,
                    "n_episodes":    n,
                    "wipe_rate":     round(wipe_rate, 4),
                    "corner_rate":   round(corner_rate, 4),
                    "concede_rate":  round(concede_rate, 4),
                    "avg_duration":  round(avg_duration, 2),
                    "avg_alliances": round(avg_alliances, 2),
                    "avg_reward":    round(avg_reward, 4),
                })

        # Reset episode buffers
        self._episode_outcomes  = []
        self._episode_durations = []
        self._episode_alliances = []
        self._episode_rewards   = []


# ── Training Entry Point ───────────────────────────────────────────────────

def run_exp_jh001(
    total_timesteps: int = 500_000,
    seed: int = 42,
    log_interval: int = 5000,
    save_dir: str = "experiments/working",
):
    print("=" * 65)
    print("  EXP-JH001: Jack PPO vs Rule-Based Regular Players")
    print(f"  timesteps={total_timesteps:,} | seed={seed}")
    print(f"  Network: MlpPolicy 256x256")
    print(f"  Research question: What strategy does the Jack")
    print(f"  agent discover against rule-based opponents?")
    print("=" * 65)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("docs/screenshots", exist_ok=True)

    timestamp  = time.strftime("%Y%m%d")
    csv_path   = os.path.join(
        save_dir, f"EXP_JH001_log_{timestamp}.csv")
    model_path = os.path.join(
        save_dir, "EXP_JH001_jack_ppo")

    # Build environment
    env = JackTrainingEnv(seed=seed)

    # PPO with larger network for richer observation space
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
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=seed,
    )

    callback = JackTrainingCallback(
        log_interval=log_interval,
        csv_path=csv_path,
        verbose=0,
    )

    print(f"\n  Training started — logging every "
          f"{log_interval:,} steps\n")
    start = time.time()
    model.learn(total_timesteps=total_timesteps,
                callback=callback)
    elapsed = time.time() - start
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")

    # Final evaluation — 100 games, deterministic
    print("\n  Running final evaluation (100 games)...")
    outcomes   = []
    durations  = []
    alliances  = []

    for game_num in range(100):
        obs, _ = env.reset()
        game_done = False
        rounds = 0
        while not game_done and rounds < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            game_done = terminated or truncated
            rounds += 1
        outcomes.append(info.get("outcome", "timeout"))
        durations.append(rounds)
        alliances.append(info.get("n_alliances", 0))

    wipe_rate    = outcomes.count("jack_wins_lobby_wipe") / 100
    corner_rate  = outcomes.count(
        "regular_wins_suspicion_consensus") / 100
    concede_rate = outcomes.count(
        "jack_voluntary_concession") / 100
    avg_duration = np.mean(durations)
    avg_alliances = np.mean(alliances)

    model.save(model_path)

    print("\n" + "=" * 65)
    print("  EXP-JH001 FINAL EVALUATION (100 games, deterministic)")
    print("=" * 65)
    print(f"  Lobby wipe rate    : {wipe_rate:.2f}")
    print(f"  Jack cornered rate : {corner_rate:.2f}")
    print(f"  Concession rate    : {concede_rate:.2f}")
    print(f"  Avg game duration  : {avg_duration:.1f} rounds")
    print(f"  Avg alliances      : {avg_alliances:.2f}")
    print(f"  Model saved        : {model_path}.zip")
    print(f"  Log saved          : {csv_path}")
    print("=" * 65)

    return {
        "wipe_rate":     wipe_rate,
        "corner_rate":   corner_rate,
        "concede_rate":  concede_rate,
        "avg_duration":  avg_duration,
        "avg_alliances": avg_alliances,
    }


if __name__ == "__main__":
    run_exp_jh001(
        total_timesteps=500_000,
        seed=42,
        log_interval=5000,
        save_dir="experiments/working",
    )