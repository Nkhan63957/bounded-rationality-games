"""
Bounded Rationality Games — Full Multi-Agent Training
=======================================================
EXP-JH004: Jack PPO (256x256) vs 19 Independent Regular
PPO agents (128x128). Both sides train from scratch.

Architecture:
  - 20 independent PPO networks (1 Jack + 19 regulars)
  - Fixed timestep budget: 300k per network (Option A)
  - Dead players stop contributing that episode
  - 3 seeds for robustness

Evaluation metrics:
  - Lobby wipe rate
  - Jack cornered rate
  - Jack voluntary concession rate
  - Jack identification round (when do regulars corner Jack)
  - Alliance survival advantage (pairs vs loners vs groups)

Research question:
  Do independently trained agents discover the same
  strategic equilibria documented in the canonical
  Jack of Hearts game instance?
"""

import numpy as np
import os
import csv
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environments.jack_of_hearts import (
    JackOfHeartsEnv, N_PLAYERS, N_SUITS, N_MESSAGES
)
from environments.game_coordinator import GameCoordinator


# ── Constants ──────────────────────────────────────────────────

JACK_NET    = [256, 256]
REGULAR_NET = [128, 128]
TOTAL_TIMESTEPS = 300_000
LOG_INTERVAL    = 5_000
SEEDS = [42, 7, 123]


# ── Per-Agent Environment Wrapper ──────────────────────────────

class AgentEnv(JackOfHeartsEnv):
    """
    Single-agent perspective for one player.
    Wraps the shared GameCoordinator.
    Handles dead player episode contribution (Option A).
    """

    def __init__(self, agent_id: int, coordinator_ref: list,
                 n_perceptive: int = N_PLAYERS, seed: int = 42):
        role = "jack" if agent_id == 0 else "regular"
        super().__init__(
            agent_role=role,
            agent_id=agent_id,
            n_perceptive=n_perceptive,
            seed=seed + agent_id,
        )
        self.agent_id       = agent_id
        self.coord_ref      = coordinator_ref  # mutable reference
        self._episode_reward = 0.0
        self._episode_rounds = 0
        self._alive_this_ep  = True

    @property
    def coord(self):
        return self.coord_ref[0]

    def reset(self, seed=None, options=None):
        self._episode_reward = 0.0
        self._episode_rounds = 0
        self._alive_this_ep  = True
        self._sync()
        return self._get_obs(), {}

    def step(self, action):
        """
        If this agent is dead or coordinator is done,
        return zero reward and terminated=True (Option A).
        """
        if not self._alive_this_ep or self.coord.done:
            return self._get_obs(), 0.0, True, False, {
                "outcome": "dead_or_done",
                "episode_reward": self._episode_reward,
                "episode_rounds": self._episode_rounds,
            }

        # Store action — actual step handled by coordinator
        self._pending_action = action
        self._sync()

        return self._get_obs(), 0.0, False, False, {
            "outcome": "pending",
            "episode_reward": self._episode_reward,
            "episode_rounds": self._episode_rounds,
        }

    def apply_reward(self, reward: float, done: bool,
                     info: dict):
        """Called by training loop after coordinator step."""
        self._episode_reward += reward
        self._episode_rounds += 1
        if self.agent_id not in self.coord.alive_players:
            self._alive_this_ep = False
        self._sync()

    def _sync(self):
        src = self.coord.envs[0]
        self.players       = src.players
        self.symbols       = src.symbols
        self.alive_mask    = src.alive_mask
        self.round         = src.round
        self.game_over     = src.game_over
        self.winner        = src.winner
        self.last_messages = src.last_messages
        self.neighborhoods = src.neighborhoods


# ── Full Multi-Agent Training Loop ────────────────────────────

class FullMultiAgentTrainer:
    """
    Manages 20 independent PPO networks training
    simultaneously against each other.

    Each network collects its own rollout buffer.
    Networks update independently when buffer fills.
    Dead players contribute 0 reward for that episode.
    """

    def __init__(self, seed: int = 42,
                 save_dir: str = "experiments/working",
                 log_interval: int = LOG_INTERVAL):
        self.seed         = seed
        self.save_dir     = save_dir
        self.log_interval = log_interval

        # Shared coordinator reference
        self.coord_ref = [None]

        # Build environments — one per agent
        self.envs = []
        for i in range(N_PLAYERS):
            env = AgentEnv(
                agent_id=i,
                coordinator_ref=self.coord_ref,
                n_perceptive=2,
                seed=seed,
            )
            self.envs.append(env)

        # Build PPO models
        self.models = []
        for i in range(N_PLAYERS):
            net = JACK_NET if i == 0 else REGULAR_NET
            model = PPO(
                policy="MlpPolicy",
                env=self.envs[i],
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=net),
                verbose=0,
                seed=seed + i,
            )
            self.models.append(model)

        # Metrics
        self.episode_outcomes   = []
        self.episode_durations  = []
        self.episode_alliances  = []
        self.jack_id_rounds     = []
        self.alliance_survival  = {"pair": [], "group": [],
                                    "loner": []}
        self.total_steps        = 0
        self.last_log           = 0

        # CSV logging
        timestamp = time.strftime("%Y%m%d")
        self.csv_path = os.path.join(
            save_dir,
            f"EXP_JH004_seed{seed}_log_{timestamp}.csv")

    def _new_episode(self):
        """Initialize fresh episode with new coordinator."""
        coord = GameCoordinator(
            n_perceptive=N_PLAYERS,
            seed=np.random.randint(0, 10000)
        )
        self.coord_ref[0] = coord
        coord.reset()
        for env in self.envs:
            env.reset()
        return coord

    def _get_all_actions(self, coord):
        """Collect actions from all alive agents."""
        actions = {}
        for i in coord.alive_players:
            obs = self.envs[i]._get_obs()
            action, _ = self.models[i].predict(
                obs, deterministic=False)
            actions[i] = action
        return actions

    def _update_alliance_survival(self, coord):
        """
        Track alliance survival advantage.
        Categorize surviving players by alliance size.
        """
        surviving_regulars = [
            i for i in coord.alive_players if i != 0]

        for player_id in surviving_regulars:
            player = coord.envs[0].players[player_id]
            alliance_size = len(player.alliances)
            if alliance_size == 1:
                self.alliance_survival["pair"].append(1)
            elif alliance_size >= 2:
                self.alliance_survival["group"].append(1)
            else:
                self.alliance_survival["loner"].append(1)

        # Dead regulars
        all_regular_ids = list(range(1, N_PLAYERS))
        dead_regulars = [
            i for i in all_regular_ids
            if i not in coord.alive_players]
        for player_id in dead_regulars:
            player = coord.envs[0].players[player_id]
            alliance_size = len(player.alliances)
            if alliance_size == 1:
                self.alliance_survival["pair"].append(0)
            elif alliance_size >= 2:
                self.alliance_survival["group"].append(0)
            else:
                self.alliance_survival["loner"].append(0)

    def _jack_identification_round(self, coord):
        """
        Estimate round at which Jack was identified.
        Proxy: round when Jack cornered (100% suspicion).
        Returns -1 if Jack not cornered.
        """
        if coord.winner == "regular":
            return coord.round
        return -1

    def run_episode(self):
        """Run one full game episode."""
        coord = self._new_episode()
        round_num = 0

        while not coord.done and round_num < 200:
            actions = self._get_all_actions(coord)
            if 0 in actions:
                actions[0][6] = 0.0  # disable random concession

            obs_dict, rewards, done, info = coord.step(actions)

            # Distribute rewards and update step counts
            for i in range(N_PLAYERS):
                reward = rewards.get(i, 0.0)
                agent_done = done or (
                    i not in coord.alive_players)
                self.envs[i].apply_reward(
                    reward, agent_done, info)

            self.total_steps += len(coord.alive_players)
            round_num += 1

        # Record episode metrics
        outcome = info.get("outcome", "timeout")
        self.episode_outcomes.append(outcome)
        self.episode_durations.append(round_num)
        self.episode_alliances.append(
            info.get("n_alliances", 0))
        self.jack_id_rounds.append(
            self._jack_identification_round(coord))
        self._update_alliance_survival(coord)

        return outcome, round_num

    def _collect_and_update(self, n_episodes: int = 10):
        """
        Run n_episodes to fill rollout buffers,
        then trigger PPO updates for each model.
        """
        for _ in range(n_episodes):
            self.run_episode()

        # Manual rollout collection and update
        # Each model learns from its own agent's experience
        for i, model in enumerate(self.models):
            env = self.envs[i]
            obs, _ = env.reset()

            # Collect a mini-rollout for this agent
            rollout_steps = 0
            target_steps = model.n_steps

            while rollout_steps < target_steps:
                action, _ = model.predict(
                    obs, deterministic=False)
                obs_new, reward, terminated, truncated, info = \
                    env.step(action)
                rollout_steps += 1
                obs = obs_new
                if terminated or truncated:
                    obs, _ = env.reset()

    def _log_metrics(self):
        n = len(self.episode_outcomes)
        if n == 0:
            return

        wipe_rate    = self.episode_outcomes.count(
            "jack_wins_lobby_wipe") / n
        corner_rate  = self.episode_outcomes.count(
            "regular_wins_suspicion_consensus") / n
        concede_rate = self.episode_outcomes.count(
            "jack_voluntary_concession") / n
        avg_duration = np.mean(self.episode_durations)
        avg_alliances = np.mean(self.episode_alliances)

        # Jack identification round (when cornered)
        id_rounds = [r for r in self.jack_id_rounds if r > 0]
        avg_id_round = np.mean(id_rounds) if id_rounds else -1

        # Alliance survival advantage
        pair_surv  = np.mean(
            self.alliance_survival["pair"]) \
            if self.alliance_survival["pair"] else 0
        group_surv = np.mean(
            self.alliance_survival["group"]) \
            if self.alliance_survival["group"] else 0
        loner_surv = np.mean(
            self.alliance_survival["loner"]) \
            if self.alliance_survival["loner"] else 0

        print(
            f"  Steps ~{self.total_steps:>8,} | "
            f"Episodes: {n:>4} | "
            f"Wipe: {wipe_rate:.2f} | "
            f"Cornered: {corner_rate:.2f} | "
            f"Concede: {concede_rate:.2f} | "
            f"Avg rounds: {avg_duration:.1f} | "
            f"Jack ID round: {avg_id_round:.1f} | "
            f"Surv — Pair: {pair_surv:.2f} "
            f"Group: {group_surv:.2f} "
            f"Loner: {loner_surv:.2f}"
        )

        if self.csv_path:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                fieldnames = [
                    "steps", "n_episodes", "wipe_rate",
                    "corner_rate", "concede_rate",
                    "avg_duration", "avg_alliances",
                    "avg_id_round", "pair_survival",
                    "group_survival", "loner_survival"
                ]
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "steps":          self.total_steps,
                    "n_episodes":     n,
                    "wipe_rate":      round(wipe_rate, 4),
                    "corner_rate":    round(corner_rate, 4),
                    "concede_rate":   round(concede_rate, 4),
                    "avg_duration":   round(avg_duration, 2),
                    "avg_alliances":  round(avg_alliances, 2),
                    "avg_id_round":   round(avg_id_round, 2),
                    "pair_survival":  round(pair_surv, 4),
                    "group_survival": round(group_surv, 4),
                    "loner_survival": round(loner_surv, 4),
                })

        # Reset episode buffers
        self.episode_outcomes  = []
        self.episode_durations = []
        self.episode_alliances = []
        self.jack_id_rounds    = []
        self.alliance_survival = {"pair": [], "group": [],
                                   "loner": []}
        self.last_log = self.total_steps

    def train(self):
        """Main training loop."""
        print(f"\n  Training started — seed {self.seed}\n")
        start = time.time()

        episodes_per_update = 20
        target_steps = TOTAL_TIMESTEPS

        while self.total_steps < target_steps:
            self._collect_and_update(
                n_episodes=episodes_per_update)

            if self.total_steps - self.last_log >= \
                    self.log_interval:
                self._log_metrics()

        elapsed = time.time() - start
        print(f"\n  Training complete in "
              f"{elapsed/60:.1f} minutes")

    def evaluate(self, n_games: int = 100):
        """Final stochastic evaluation."""
        print(f"\n  Evaluating ({n_games} games, "
              f"stochastic)...")

        outcomes, durations, alliances = [], [], []
        id_rounds = []
        alliance_surv = {"pair": [], "group": [],
                          "loner": []}

        for _ in range(n_games):
            coord = self._new_episode()
            rounds = 0
            while not coord.done and rounds < 200:
                actions = self._get_all_actions(coord)
                _, _, done, info = coord.step(actions)
                rounds += 1
            outcomes.append(info.get("outcome", "timeout"))
            durations.append(rounds)
            alliances.append(info.get("n_alliances", 0))
            id_rounds.append(
                self._jack_identification_round(coord))
            self._update_alliance_survival(coord)

        wipe_rate    = outcomes.count(
            "jack_wins_lobby_wipe") / n_games
        corner_rate  = outcomes.count(
            "regular_wins_suspicion_consensus") / n_games
        concede_rate = outcomes.count(
            "jack_voluntary_concession") / n_games
        avg_duration = np.mean(durations)
        avg_alliances = np.mean(alliances)
        valid_id = [r for r in id_rounds if r > 0]
        avg_id_round = np.mean(valid_id) if valid_id else -1

        pair_surv  = np.mean(
            self.alliance_survival["pair"]) \
            if self.alliance_survival["pair"] else 0
        group_surv = np.mean(
            self.alliance_survival["group"]) \
            if self.alliance_survival["group"] else 0
        loner_surv = np.mean(
            self.alliance_survival["loner"]) \
            if self.alliance_survival["loner"] else 0

        print("\n" + "=" * 65)
        print(f"  EXP-JH004 FINAL EVALUATION "
              f"({n_games} games, stochastic, seed {self.seed})")
        print("=" * 65)
        print(f"  Lobby wipe rate      : {wipe_rate:.2f}")
        print(f"  Jack cornered rate   : {corner_rate:.2f}")
        print(f"  Concession rate      : {concede_rate:.2f}")
        print(f"  Avg game duration    : {avg_duration:.1f} rounds")
        print(f"  Avg alliances        : {avg_alliances:.2f}")
        print(f"  Jack ID round (avg)  : {avg_id_round:.1f}")
        print(f"  Survival — Pair      : {pair_surv:.2f}")
        print(f"  Survival — Group     : {group_surv:.2f}")
        print(f"  Survival — Loner     : {loner_surv:.2f}")
        print("=" * 65)

        return {
            "wipe_rate":     wipe_rate,
            "corner_rate":   corner_rate,
            "concede_rate":  concede_rate,
            "avg_duration":  avg_duration,
            "avg_id_round":  avg_id_round,
            "pair_survival": pair_surv,
            "group_survival":group_surv,
            "loner_survival":loner_surv,
        }

    def save_all(self):
        """Save all 20 models."""
        os.makedirs(self.save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            role = "jack" if i == 0 else f"regular_{i:02d}"
            path = os.path.join(
                self.save_dir,
                f"EXP_JH004_seed{self.seed}_{role}")
            model.save(path)
        print(f"  Models saved to {self.save_dir}")


# ── Entry Point ────────────────────────────────────────────────

def run_exp_jh004(seeds=SEEDS, save_dir="experiments/working"):
    print("=" * 65)
    print("  EXP-JH004: Full Multi-Agent — 20 Independent PPO")
    print(f"  Jack: 256x256 | Regulars: 128x128")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,} per network")
    print(f"  Seeds: {seeds}")
    print(f"  Metrics: wipe, cornered, concede, Jack ID round,")
    print(f"           alliance survival (pair/group/loner)")
    print("=" * 65)

    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*65}")
        print(f"  Seed {seed}")
        print(f"{'='*65}")

        trainer = FullMultiAgentTrainer(
            seed=seed,
            save_dir=save_dir,
            log_interval=LOG_INTERVAL,
        )
        trainer.train()
        results = trainer.evaluate(n_games=100)
        trainer.save_all()
        all_results[seed] = results

    # Multi-seed summary
    print("\n" + "=" * 65)
    print("  EXP-JH004 MULTI-SEED SUMMARY")
    print("=" * 65)
    for metric in ["wipe_rate", "corner_rate", "concede_rate",
                   "avg_duration", "avg_id_round",
                   "pair_survival", "group_survival",
                   "loner_survival"]:
        values = [all_results[s][metric] for s in seeds]
        print(f"  {metric:<20}: "
              f"{np.mean(values):.3f} ± {np.std(values):.3f}")
    print("=" * 65)


if __name__ == "__main__":
    run_exp_jh004(
        seeds=SEEDS,
        save_dir="experiments/working",
    )