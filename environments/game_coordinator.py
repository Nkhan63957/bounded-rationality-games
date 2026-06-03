"""
Jack of Hearts — Game Coordinator (v2)
========================================
Manages all 20 agents simultaneously with full alliance
and manipulation mechanics.

Step flow per round:
  1. Collect actions from all alive agents
  2. Process alliance offers — mutual offers form alliances
  3. Jack manipulation — diminishing returns brainwashing
  4. Process messages — update suspicion distributions
  5. Process declarations — eliminate wrong declarers
  6. Check win conditions
  7. Distribute rewards
  8. Update symbol beliefs from neighborhoods
  9. Return observations for next round
"""

import numpy as np
from environments.jack_of_hearts import (
    JackOfHeartsEnv, N_PLAYERS, N_SUITS, N_MESSAGES,
    JACK_WIN_REWARD, JACK_LOSE_REWARD, JACK_ELIM_REWARD,
    REGULAR_WIN_REWARD, REGULAR_LOSE_REWARD, REGULAR_SURV_REWARD,
    ALLIANCE_SURV_BONUS, SUSPICION_THRESHOLD,
    MANIP_BASE_EFFECT, MANIP_DECAY,
)


class GameCoordinator:
    """
    Coordinates all 20 JackOfHeartsEnv instances.

    All environments share the same underlying game state.
    The coordinator is the single source of truth for:
      - alive_mask
      - player states (suspicion, alliances, beliefs)
      - round number
      - win condition
    """

    def __init__(self, n_perceptive: int = 2, seed: int = 42):
        self.n_perceptive = n_perceptive
        self.rng = np.random.default_rng(seed)

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

        # Metrics tracked across game
        self.lobby_wipe_count    = 0
        self.jack_cornered_count = 0
        self.jack_concede_count  = 0
        self.max_round_reached   = 0

    # ── Reset ──────────────────────────────────────────────────

    def reset(self):
        self.done = False
        self.round = 0
        self.winner = None
        self.alive_players = list(range(N_PLAYERS))

        obs = {}
        for i, env in enumerate(self.envs):
            o, _ = env.reset()
            obs[i] = o

        self._sync_state()
        return obs

    def _sync_state(self):
        """Propagate shared state from env 0 to all others."""
        src = self.envs[0]
        for env in self.envs[1:]:
            env.players       = src.players
            env.symbols       = src.symbols
            env.alive_mask    = src.alive_mask
            env.round         = src.round
            env.game_over     = src.game_over
            env.winner        = src.winner
            env.last_messages = src.last_messages
            env.neighborhoods = src.neighborhoods
            env.pending_offers = src.pending_offers

    # ── Step ───────────────────────────────────────────────────

    def step(self, actions: dict):
        """
        Process one full round for all alive agents.

        Parameters
        ----------
        actions : dict {agent_id: np.ndarray shape (8,)}

        Returns
        -------
        obs     : dict of observations
        rewards : dict of rewards
        done    : bool
        info    : dict of game metrics
        """
        if self.done:
            raise RuntimeError("Game over. Call reset().")

        rewards = {i: 0.0 for i in range(N_PLAYERS)}
        env0 = self.envs[0]

        # ── Phase 1: Decode all actions ────────────────────────
        decoded = {}
        for agent_id in self.alive_players:
            if agent_id not in actions:
                continue
            a = actions[agent_id]
            decoded[agent_id] = {
                "msg_type":      int(np.clip(int(a[0]*(N_MESSAGES-1)+0.5), 0, N_MESSAGES-1)),
                "msg_target":    int(np.clip(int(a[1]*(N_PLAYERS-1)+0.5), 0, N_PLAYERS-1)),
                "declaration":   int(np.clip(int(a[2]*(N_SUITS-1)+0.5),   0, N_SUITS-1)),
                "broadcast_sym": int(np.clip(int(a[3]*(N_SUITS-1)+0.5),   0, N_SUITS-1)),
                "alliance_offer":float(a[4]) > 0.5,
                "alliance_tgt":  int(np.clip(int(a[5]*(N_PLAYERS-1)+0.5), 0, N_PLAYERS-1)),
                "concede":       float(a[6]) > 0.85,
                "manip_tgt":     int(np.clip(int(a[7]*(N_PLAYERS-1)+0.5), 0, N_PLAYERS-1)),
            }
            # Jack always declares true symbol
            if agent_id == 0:
                decoded[agent_id]["declaration"] = int(
                    env0.players[0].true_symbol)

        # ── Phase 2: Jack voluntary concession ────────────────
        if 0 in decoded and decoded[0]["concede"]:
            self.done = True
            self.winner = "regular"
            self.jack_concede_count += 1
            rewards[0] += JACK_LOSE_REWARD
            for i in self.alive_players:
                if i != 0:
                    rewards[i] += REGULAR_WIN_REWARD
            return self._get_all_obs(), rewards, True, self._info(
                "jack_voluntary_concession")

        # ── Phase 3: Alliance formation ────────────────────────
        # Collect all offers this round
        offers_this_round = {}
        for agent_id, d in decoded.items():
            if d["alliance_offer"]:
                tgt = d["alliance_tgt"]
                if tgt != agent_id and tgt in self.alive_players:
                    offers_this_round[agent_id] = tgt

        # Mutual offers → form alliance
        for offerer, target in offers_this_round.items():
            if offers_this_round.get(target) == offerer:
                # Mutual — form alliance
                env0.players[offerer].alliances.add(target)
                env0.players[target].alliances.add(offerer)

        # Remove dead players from alliances
        for p in env0.players:
            p.alliances = {a for a in p.alliances if a in self.alive_players}

        # ── Phase 4: Jack manipulation ─────────────────────────
        if 0 in decoded:
            jack = env0.players[0]
            tgt = decoded[0]["manip_tgt"]
            if tgt != 0 and tgt in self.alive_players:
                count = jack.manipulation_counts.get(tgt, 0)
                effect = MANIP_BASE_EFFECT * (MANIP_DECAY ** count)
                env0.players[tgt].update_suspicion(0, -effect)
                jack.manipulation_counts[tgt] = count + 1

        # ── Phase 5: Process messages ──────────────────────────
        for agent_id, d in decoded.items():
            msg_type   = d["msg_type"]
            msg_target = d["msg_target"]
            broadcast  = d["broadcast_sym"]

            env0.last_messages[agent_id] = (msg_type, msg_target)

            # Within-alliance symbol claims (can lie)
            if msg_type == 0:
                # Sender claims to see broadcast on msg_target's collar
                if msg_target != agent_id:
                    tgt_player = env0.players[msg_target]
                    if tgt_player.alive:
                        tgt_player.alliance_claims[agent_id] = broadcast
                        if agent_id in tgt_player.alliances:
                            tgt_player.update_own_belief(broadcast, 0.15)

            # Update suspicion for all players
            for p in env0.players:
                if not p.alive or p.player_id == agent_id:
                    continue
                if msg_type == 1:    # accusation
                    p.update_suspicion(msg_target, +0.10)
                elif msg_type == 2:  # defense
                    p.update_suspicion(msg_target, -0.05)
                elif msg_type == 3:  # denial
                    p.update_suspicion(agent_id, -0.05)

        # ── Phase 6: Declarations ──────────────────────────────
        newly_eliminated = []
        for agent_id in self.alive_players:
            if agent_id not in decoded:
                continue
            player = env0.players[agent_id]
            declaration = decoded[agent_id]["declaration"]
            true_sym    = player.true_symbol
            survived    = (declaration == true_sym)

            if not survived:
                newly_eliminated.append(agent_id)
                player.alive = False
                env0.alive_mask[agent_id] = 0.0
                rewards[agent_id] += REGULAR_LOSE_REWARD
                # Jack gets elimination reward
                rewards[0] += JACK_ELIM_REWARD
                # Alliance partners lose survival bonus
                for ally_id in player.alliances:
                    rewards[ally_id] -= ALLIANCE_SURV_BONUS
            else:
                if agent_id != 0:
                    rewards[agent_id] += REGULAR_SURV_REWARD
                    # Alliance partners get survival bonus
                    for ally_id in player.alliances:
                        if ally_id in self.alive_players:
                            rewards[ally_id] += ALLIANCE_SURV_BONUS

        # Remove eliminated players
        self.alive_players = [
            i for i in self.alive_players
            if i not in newly_eliminated
        ]

        # ── Phase 7: Update symbol beliefs ────────────────────
        env0._update_all_symbol_beliefs()

        # ── Phase 8: Win conditions ────────────────────────────
        n_alive_regular = sum(
            1 for i in self.alive_players if i != 0)

        # Jack wins: lobby wipe
        if n_alive_regular == 0:
            self.done = True
            self.winner = "jack"
            self.lobby_wipe_count += 1
            rewards[0] += JACK_WIN_REWARD
            return self._get_all_obs(), rewards, True, self._info(
                "jack_wins_lobby_wipe")

        # Regular players win: 100% suspicion consensus
        if 0 in self.alive_players:
            jack_suspicion = np.array([
                env0.players[i].suspicion[0]
                for i in self.alive_players if i != 0
            ])
            if len(jack_suspicion) > 0 and \
                    (jack_suspicion >= SUSPICION_THRESHOLD).all():
                self.done = True
                self.winner = "regular"
                self.jack_cornered_count += 1
                rewards[0] += JACK_LOSE_REWARD
                for i in self.alive_players:
                    if i != 0:
                        rewards[i] += REGULAR_WIN_REWARD
                return self._get_all_obs(), rewards, True, self._info(
                    "regular_wins_suspicion_consensus")

        # ── Phase 9: Advance round ─────────────────────────────
        self.round += 1
        env0.round = self.round
        self.max_round_reached = max(self.max_round_reached, self.round)
        self._sync_state()

        return self._get_all_obs(), rewards, False, self._info("ongoing")

    # ── Helpers ────────────────────────────────────────────────

    def _get_all_obs(self):
        obs = {}
        src = self.envs[0]
        for i in range(N_PLAYERS):
            self.envs[i].round         = src.round
            self.envs[i].alive_mask    = src.alive_mask
            self.envs[i].last_messages = src.last_messages
            self.envs[i].players       = src.players
            obs[i] = self.envs[i]._get_obs()
        return obs

    def _info(self, outcome: str) -> dict:
        n_alive_regular = sum(
            1 for i in self.alive_players if i != 0)
        n_alliances = sum(
            len(self.envs[0].players[i].alliances)
            for i in self.alive_players
        ) // 2  # each alliance counted twice

        return {
            "outcome":          outcome,
            "round":            self.round,
            "alive":            len(self.alive_players),    
            "n_alive_regular":  n_alive_regular,
            "n_alliances":      n_alliances,
            "winner":           self.winner,
            "lobby_wipes":      self.lobby_wipe_count,
            "jack_cornered":    self.jack_cornered_count,
            "jack_concessions": self.jack_concede_count,
        }

    @property
    def jack_alive(self) -> bool:
        return 0 in self.alive_players