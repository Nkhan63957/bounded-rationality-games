"""
Jack of Hearts — Game Coordinator
===================================
Manages all 20 agents simultaneously.
Collects actions, processes declarations,
eliminates players, checks win conditions.

Usage:
    coordinator = GameCoordinator()
    obs = coordinator.reset()
    while not coordinator.done:
        actions = {i: policy_i.predict(obs[i]) 
                   for i in coordinator.alive_players}
        obs, rewards, done, info = coordinator.step(actions)
"""

import numpy as np
from environments.jack_of_hearts import (
    JackOfHeartsEnv, N_PLAYERS, N_SUITS, N_MESSAGES,
    JACK_WIN_REWARD, JACK_LOSE_REWARD, JACK_ELIM_REWARD,
    REGULAR_WIN_REWARD, REGULAR_LOSE_REWARD, REGULAR_SURV_REWARD,
    SUSPICION_THRESHOLD
)


class GameCoordinator:
    """
    Coordinates all 20 JackOfHeartsEnv instances.
    
    Each agent has its own environment instance sharing
    the same underlying game state via this coordinator.
    
    Step flow per round:
      1. Collect actions from all alive agents
      2. Process messages — update all suspicion distributions
      3. Process declarations — eliminate wrong declarers
      4. Check win conditions
      5. Distribute rewards
      6. Return observations for next round
    """

    def __init__(self, n_perceptive: int = 2, seed: int = 42):
        self.n_perceptive = n_perceptive
        self.rng = np.random.default_rng(seed)

        # One env per agent
        self.envs = []
        for i in range(N_PLAYERS):
            role = "jack" if i == 0 else "regular"
            env = JackOfHeartsEnv(
                agent_role=role,
                agent_id=i,
                n_perceptive=n_perceptive,
                seed=seed + i,
            )
            self.envs.append(env)

        self.done = False
        self.round = 0
        self.winner = None
        self.alive_players = list(range(N_PLAYERS))

    def reset(self):
        """Reset all environments, return initial observations."""
        self.done = False
        self.round = 0
        self.winner = None

        # Reset all envs — env 0 initializes game state,
        # others sync to it
        obs = {}
        for i, env in enumerate(self.envs):
            o, _ = env.reset()
            obs[i] = o

        # Sync shared state across all envs
        self._sync_state()

        self.alive_players = list(range(N_PLAYERS))
        return obs

    def _sync_state(self):
        """Copy game state from env 0 to all others."""
        src = self.envs[0]
        for env in self.envs[1:]:
            env.players    = src.players
            env.symbols    = src.symbols
            env.alive_mask = src.alive_mask
            env.round      = src.round
            env.game_over  = src.game_over
            env.winner     = src.winner
            env.last_messages = src.last_messages

    def step(self, actions: dict):
        """
        Process one round for all alive agents.

        Parameters
        ----------
        actions : dict
            {agent_id: action_array} for all alive agents

        Returns
        -------
        obs     : dict of observations
        rewards : dict of rewards
        done    : bool
        info    : dict
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset().")

        rewards = {i: 0.0 for i in range(N_PLAYERS)}
        env0 = self.envs[0]

        # ── Phase 1: Process messages ──────────────────────────
        # All alive agents broadcast messages simultaneously
        messages_this_round = {}
        for agent_id in self.alive_players:
            if agent_id not in actions:
                continue
            action = actions[agent_id]
            msg_type = int(np.clip(
                int(action[0] * (N_MESSAGES - 1) + 0.5), 0, N_MESSAGES - 1))
            msg_target = int(np.clip(
                int(action[1] * (N_PLAYERS - 1) + 0.5), 0, N_PLAYERS - 1))
            messages_this_round[agent_id] = (msg_type, msg_target)

        # Update last_messages on shared state
        for agent_id, msg in messages_this_round.items():
            env0.last_messages[agent_id] = msg

        # Update suspicion based on all messages
        for agent_id, (msg_type, msg_target) in messages_this_round.items():
            for p in env0.players:
                if not p.alive or p.player_id == agent_id:
                    continue
                if msg_type == 1:    # accusation
                    p.update_suspicion(msg_target, +0.1)
                elif msg_type == 2:  # defense
                    p.update_suspicion(msg_target, -0.05)
                elif msg_type == 3:  # denial
                    p.update_suspicion(agent_id, -0.05)

        # ── Phase 2: Jack voluntary concession ────────────────
        jack_action = actions.get(0)
        if jack_action is not None:
            if float(jack_action[3]) > 0.85 and 0 in self.alive_players:
                self.done = True
                self.winner = "regular"
                rewards[0] += JACK_LOSE_REWARD
                for i in self.alive_players:
                    if i != 0:
                        rewards[i] += REGULAR_WIN_REWARD
                return self._get_all_obs(), rewards, True, {
                    "outcome": "jack_voluntary_concession",
                    "round": self.round,
                    "alive": len(self.alive_players),
                }

        # ── Phase 3: Process declarations ─────────────────────
        newly_eliminated = []
        for agent_id in self.alive_players:
            if agent_id not in actions:
                continue
            action = actions[agent_id]
            player = env0.players[agent_id]

            # Jack always knows his symbol
            if agent_id == 0:
                declaration = int(player.true_symbol)
            else:
                declaration = int(np.clip(
                    int(action[2] * (N_SUITS - 1) + 0.5), 0, N_SUITS - 1))

            true_sym = player.true_symbol
            survived = (declaration == true_sym)

            if not survived:
                newly_eliminated.append(agent_id)
                player.alive = False
                env0.alive_mask[agent_id] = 0.0
                if agent_id == 0:
                    # Jack eliminated (shouldn't happen but handle)
                    rewards[agent_id] += JACK_LOSE_REWARD
                else:
                    rewards[agent_id] += REGULAR_LOSE_REWARD
                    rewards[0] += JACK_ELIM_REWARD
            else:
                if agent_id != 0:
                    rewards[agent_id] += REGULAR_SURV_REWARD

        # Remove eliminated from alive list
        self.alive_players = [
            i for i in self.alive_players
            if i not in newly_eliminated
        ]

        # ── Phase 4: Check win conditions ─────────────────────
        n_alive_regular = sum(
            1 for i in self.alive_players if i != 0)

        # Jack wins
        if n_alive_regular == 0:
            self.done = True
            self.winner = "jack"
            rewards[0] += JACK_WIN_REWARD
            info = {
                "outcome": "jack_wins_lobby_wipe",
                "round": self.round,
                "alive": len(self.alive_players),
            }
            return self._get_all_obs(), rewards, True, info

        # Regular players win: 100% suspicion consensus
        jack_suspicion = np.array([
            env0.players[i].suspicion[0]
            for i in self.alive_players if i != 0
        ])
        if len(jack_suspicion) > 0 and \
           (jack_suspicion >= SUSPICION_THRESHOLD).all():
            self.done = True
            self.winner = "regular"
            rewards[0] += JACK_LOSE_REWARD
            for i in self.alive_players:
                if i != 0:
                    rewards[i] += REGULAR_WIN_REWARD
            info = {
                "outcome": "regular_wins_suspicion_consensus",
                "round": self.round,
                "alive": len(self.alive_players),
            }
            return self._get_all_obs(), rewards, True, info

        # ── Phase 5: Advance round ─────────────────────────────
        self.round += 1
        env0.round = self.round
        self._sync_state()

        return self._get_all_obs(), rewards, False, {
            "outcome": "ongoing",
            "round": self.round,
            "alive": len(self.alive_players),
            "n_regular_alive": n_alive_regular,
        }

    def _get_all_obs(self):
        """Get observations for all agents."""
        obs = {}
        for i in range(N_PLAYERS):
            self.envs[i].round      = self.envs[0].round
            self.envs[i].alive_mask = self.envs[0].alive_mask
            self.envs[i].last_messages = self.envs[0].last_messages
            obs[i] = self.envs[i]._get_obs()
        return obs