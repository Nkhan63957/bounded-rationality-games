"""
Bounded Rationality Games — Regular Player Training
=====================================================
EXP-JH004b: 19 Independent Regular PPO agents (128x128)
trained against pre-trained Jack (EXP-JH003).

Architecture:
  - Jack: fixed pre-trained policy (EXP-JH003)
  - Regulars: 19 independent PPO networks, 128x128
  - Each regular trains in its own env handling full games
  - Other regulars sampled from their current policies
  - 300k timesteps per network, 3 seeds

Research questions:
  - Do regulars learn to identify the Jack?
  - Do pairs outperform groups and loners?
  - At what round does Jack identification occur?
  - What survival rate emerges against competent Jack?
"""

import numpy as np
import os
import csv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environments.jack_of_hearts import (
    JackOfHeartsEnv, N_PLAYERS, N_SUITS, N_MESSAGES
)
from environments.game_coordinator import GameCoordinator


# ── Constants ──────────────────────────────────────────────────

REGULAR_NET     = [128, 128]
TOTAL_TIMESTEPS = 300_000
LOG_INTERVAL    = 5_000
SEEDS           = [42, 7, 123]
JACK_MODEL_PATH = "experiments/working/EXP_JH003_jack_ppo"


# ── Regular Player Training Environment ───────────────────────

class RegularTrainingEnv(JackOfHeartsEnv):
    """
    Single regular player's perspective for PPO training.
    Handles full game internally each episode.
    
    Jack: fixed pre-trained PPO (EXP-JH003)
    Other regulars: sampled from their current policies
    This agent: the PPO learner
    """

    def __init__(self, agent_id: int,
                 jack_model,
                 regular_models: list,
                 seed: int = 42):
        assert agent_id != 0, "agent_id 0 is the Jack"
        super().__init__(
            agent_role="regular",
            agent_id=agent_id,
            n_perceptive=N_PLAYERS,
            seed=seed + agent_id,
        )
        self.agent_id       = agent_id
        self.jack_model     = jack_model
        self.regular_models = regular_models  # list of 19 models
        self.coord          = None
        self._episode_rounds = 0
        self._episode_reward = 0.0
        self._last_info      = {}
        self._alive_this_ep  = True
        self.rng             = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        self.coord = GameCoordinator(
            n_perceptive=N_PLAYERS,
            seed=int(self.rng.integers(0, 10000))
        )
        self.coord.reset()
        self._episode_rounds = 0
        self._episode_reward = 0.0
        self._last_info      = {}
        self._alive_this_ep  = True
        self._sync()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """
        One PPO step = one full game round.
        This agent uses PPO action.
        Jack uses pre-trained model.
        Other regulars use their current models.
        """
        # Episode length cap
        if self._episode_rounds >= 200:
            self.coord.done = True
            return self._get_obs(), 0.0, True, False, {
                "outcome": "max_rounds_reached",
                "episode_rounds": self._episode_rounds,
                "episode_reward": self._episode_reward,
                "n_alliances": 0,
                "alive_at_end": int(
                    self.agent_id in self.coord.alive_players),
            }

        # If already dead or game over
        if not self._alive_this_ep or self.coord.done:
            return self._get_obs(), 0.0, True, False, {
                **self._last_info,
                "episode_rounds": self._episode_rounds,
                "episode_reward": self._episode_reward,
            }

        # Collect actions from all alive agents
        actions = {}

        for i in self.coord.alive_players:
            obs_i = self.coord.envs[i]._get_obs()

            if i == 0:
                # Jack: fixed pre-trained model
                act, _ = self.jack_model.predict(
                    obs_i, deterministic=False)
                act[6] = 0.0  # disable random concession
                actions[i] = act

            elif i == self.agent_id:
                # This learner: use PPO action
                actions[i] = action

            else:
                # Other regulars: sample from model if available,
                # otherwise random action
                model = self.regular_models[i]
                if model is not None:
                    act, _ = model.predict(
                        obs_i, deterministic=False)
                else:
                    act = self.coord.envs[i].action_space.sample()
                actions[i] = act

        # Step coordinator
        obs_dict, rewards, done, info = self.coord.step(actions)

        # Get this agent's reward
        my_reward = rewards.get(self.agent_id, 0.0)
        self._episode_reward += my_reward
        self._episode_rounds += 1
        self._last_info = info

        # Check if this agent is still alive
        if self.agent_id not in self.coord.alive_players:
            self._alive_this_ep = False

        # Sync state
        self._sync()

        # Build info dict
        info["episode_rounds"]  = self._episode_rounds
        info["episode_reward"]  = self._episode_reward
        info["alive_at_end"]    = int(
            self.agent_id in self.coord.alive_players)

        if done:
            info["terminal_observation"] = self._get_obs()

        return self._get_obs(), my_reward, done, False, info

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


# ── Logging Callback ───────────────────────────────────────────

class RegularTrainingCallback(BaseCallback):
    """
    Logs training metrics every log_interval timesteps.
    Tracks survival rate, alliance formation, game outcomes.
    """

    def __init__(self, agent_id: int,
                 log_interval: int = LOG_INTERVAL,
                 csv_path: str = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.agent_id     = agent_id
        self.log_interval = log_interval
        self.csv_path     = csv_path
        self._last_log    = 0

        self._outcomes    = []
        self._durations   = []
        self._alliances   = []
        self._survived    = []
        self._rewards     = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            outcome = info.get("outcome", "")
            if outcome and outcome != "pending":
                self._outcomes.append(outcome)
                self._durations.append(
                    info.get("episode_rounds", 0))
                self._alliances.append(
                    info.get("n_alliances", 0))
                self._survived.append(
                    info.get("alive_at_end", 0))
                self._rewards.append(
                    info.get("episode_reward", 0.0))

        if self.num_timesteps - self._last_log >= \
                self.log_interval:
            self._last_log = self.num_timesteps
            self._log_metrics()

        return True

    def _log_metrics(self):
        n = len(self._outcomes)
        if n == 0:
            return

        wipe_rate    = self._outcomes.count(
            "jack_wins_lobby_wipe") / n
        corner_rate  = self._outcomes.count(
            "regular_wins_suspicion_consensus") / n
        surv_rate    = np.mean(self._survived) if \
            self._survived else 0
        avg_duration = np.mean(self._durations)
        avg_reward   = np.mean(self._rewards)

        print(
            f"    Agent {self.agent_id:>2} | "
            f"Step {self.num_timesteps:>7,} | "
            f"Episodes: {n:>4} | "
            f"Surv: {surv_rate:.2f} | "
            f"Wipe: {wipe_rate:.2f} | "
            f"Corner: {corner_rate:.2f} | "
            f"Rounds: {avg_duration:.1f} | "
            f"Reward: {avg_reward:.2f}"
        )

        if self.csv_path:
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                fieldnames = [
                    "agent_id", "step", "n_episodes",
                    "survival_rate", "wipe_rate",
                    "corner_rate", "avg_duration",
                    "avg_reward"
                ]
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "agent_id":      self.agent_id,
                    "step":          self.num_timesteps,
                    "n_episodes":    n,
                    "survival_rate": round(surv_rate, 4),
                    "wipe_rate":     round(wipe_rate, 4),
                    "corner_rate":   round(corner_rate, 4),
                    "avg_duration":  round(avg_duration, 2),
                    "avg_reward":    round(avg_reward, 4),
                })

        self._outcomes   = []
        self._durations  = []
        self._alliances  = []
        self._survived   = []
        self._rewards    = []


# ── Alliance Survival Evaluator ────────────────────────────────

def evaluate_alliance_survival(models_dict: dict,
                                jack_model,
                                n_games: int = 100,
                                seed: int = 42):
    """
    Evaluate alliance survival advantage.
    Categorize surviving regulars by alliance size.
    Returns survival rates for pairs, groups, loners.
    """
    rng = np.random.default_rng(seed)
    pair_surv, group_surv, loner_surv = [], [], []
    jack_id_rounds = []
    outcomes = []

    for _ in range(n_games):
        coord = GameCoordinator(
            n_perceptive=N_PLAYERS,
            seed=int(rng.integers(0, 10000))
        )
        coord.reset()
        rounds = 0

        while not coord.done and rounds < 200:
            actions = {}
            for i in coord.alive_players:
                obs = coord.envs[i]._get_obs()
                if i == 0:
                    act, _ = jack_model.predict(
                        obs, deterministic=False)
                    act[6] = 0.0
                    actions[i] = act
                else:
                    model = models_dict.get(i)
                    if model:
                        act, _ = model.predict(
                            obs, deterministic=False)
                        actions[i] = act
                    else:
                        actions[i] = coord.envs[i]\
                            .action_space.sample()
            _, _, done, info = coord.step(actions)
            rounds += 1

        outcomes.append(info.get("outcome", "timeout"))

        # Jack identification round
        if coord.winner == "regular":
            jack_id_rounds.append(rounds)
        else:
            jack_id_rounds.append(-1)

        # Alliance survival
        for i in range(1, N_PLAYERS):
            player = coord.envs[0].players[i]
            survived = int(i in coord.alive_players)
            alliance_size = len(player.alliances)
            if alliance_size == 1:
                pair_surv.append(survived)
            elif alliance_size >= 2:
                group_surv.append(survived)
            else:
                loner_surv.append(survived)

    wipe_rate   = outcomes.count(
        "jack_wins_lobby_wipe") / n_games
    corner_rate = outcomes.count(
        "regular_wins_suspicion_consensus") / n_games
    valid_id    = [r for r in jack_id_rounds if r > 0]
    avg_id      = np.mean(valid_id) if valid_id else -1

    return {
        "wipe_rate":     wipe_rate,
        "corner_rate":   corner_rate,
        "avg_id_round":  avg_id,
        "pair_survival": np.mean(pair_surv) if pair_surv else 0,
        "group_survival":np.mean(group_surv) if group_surv else 0,
        "loner_survival":np.mean(loner_surv) if loner_surv else 0,
        "n_pairs":       len(pair_surv),
        "n_groups":      len(group_surv),
        "n_loners":      len(loner_surv),
    }


# ── Main Training Function ─────────────────────────────────────

def run_exp_jh004b(
    seeds=SEEDS,
    save_dir="experiments/working",
    jack_model_path=JACK_MODEL_PATH,
):
    print("=" * 65)
    print("  EXP-JH004b: 19 Independent Regular PPO")
    print(f"  Jack: fixed pre-trained ({jack_model_path})")
    print(f"  Regulars: 128x128, {TOTAL_TIMESTEPS:,} steps each")
    print(f"  Seeds: {seeds}")
    print("=" * 65)

    os.makedirs(save_dir, exist_ok=True)

    # Load pre-trained Jack
    print(f"\n  Loading Jack model: {jack_model_path}")
    jack_model = PPO.load(jack_model_path)
    print("  Jack model loaded.")

    all_seed_results = {}

    for seed in seeds:
        print(f"\n{'='*65}")
        print(f"  SEED {seed}")
        print(f"{'='*65}")

        timestamp = time.strftime("%Y%m%d")
        csv_path  = os.path.join(
            save_dir,
            f"EXP_JH004b_seed{seed}_log_{timestamp}.csv")

        # Initialize regular models with dummy env
        # (will be replaced with proper envs below)
        regular_models = [None] * N_PLAYERS  # index 0 = Jack

        # Build one PPO per regular player
        envs   = {}
        models = {}

        for agent_id in range(1, N_PLAYERS):
            env = RegularTrainingEnv(
                agent_id=agent_id,
                jack_model=jack_model,
                regular_models=regular_models,
                seed=seed,
            )
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
                policy_kwargs=dict(net_arch=REGULAR_NET),
                verbose=0,
                seed=seed + agent_id,
            )
            envs[agent_id]   = env
            models[agent_id] = model
            regular_models[agent_id] = model

        print(f"\n  Training {N_PLAYERS-1} regular agents...")
        print(f"  Logging every {LOG_INTERVAL:,} steps\n")

        start = time.time()

        # Train each regular agent sequentially
        for agent_id in range(1, N_PLAYERS):
            print(f"\n  --- Training Agent {agent_id} ---")
            callback = RegularTrainingCallback(
                agent_id=agent_id,
                log_interval=LOG_INTERVAL,
                csv_path=csv_path,
            )
            models[agent_id].learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callback,
            )

        elapsed = time.time() - start
        print(f"\n  Training complete in {elapsed/60:.1f} minutes")

        # Save all regular models
        for agent_id in range(1, N_PLAYERS):
            path = os.path.join(
                save_dir,
                f"EXP_JH004b_seed{seed}_regular_{agent_id:02d}")
            models[agent_id].save(path)
        print(f"  Models saved to {save_dir}")

        # Full evaluation
        print(f"\n  Running evaluation (100 games)...")
        results = evaluate_alliance_survival(
            models_dict=models,
            jack_model=jack_model,
            n_games=100,
            seed=seed,
        )

        print("\n" + "=" * 65)
        print(f"  EXP-JH004b EVALUATION — Seed {seed}")
        print("=" * 65)
        print(f"  Lobby wipe rate      : {results['wipe_rate']:.2f}")
        print(f"  Jack cornered rate   : {results['corner_rate']:.2f}")
        print(f"  Jack ID round (avg)  : {results['avg_id_round']:.1f}")
        print(f"  Pair survival        : {results['pair_survival']:.2f} (n={results['n_pairs']})")
        print(f"  Group survival       : {results['group_survival']:.2f} (n={results['n_groups']})")
        print(f"  Loner survival       : {results['loner_survival']:.2f} (n={results['n_loners']})")
        print("=" * 65)

        all_seed_results[seed] = results

    # Multi-seed summary
    print("\n" + "=" * 65)
    print("  EXP-JH004b MULTI-SEED SUMMARY")
    print("=" * 65)
    for metric in ["wipe_rate", "corner_rate", "avg_id_round",
                   "pair_survival", "group_survival",
                   "loner_survival"]:
        values = [all_seed_results[s][metric] for s in seeds]
        print(f"  {metric:<20}: "
              f"{np.mean(values):.3f} ± {np.std(values):.3f}")
    print("=" * 65)

    return all_seed_results


if __name__ == "__main__":
    run_exp_jh004b(
        seeds=SEEDS,
        save_dir="experiments/working",
        jack_model_path=JACK_MODEL_PATH,
    )