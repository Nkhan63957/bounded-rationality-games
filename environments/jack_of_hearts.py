"""
Jack of Hearts — Solitary Confinement Environment
===================================================
Based on Alice in Borderland Season 2, Episode 5.

Game structure:
  - 20 players: 1 Jack, 19 regular players
  - Each player wears a collar displaying one of 4 suit symbols
    visible to all OTHER players but not themselves
  - Jack has a hidden prosthetic mirror — he can see his own symbol
  - Each round: players communicate then declare their own symbol
  - Wrong or missing declaration → collar explosion → elimination
  - Jack wins: entire lobby wiped (all regular players dead)
  - Regular players win: Jack cornered (100% suspicion) → forced
    concession, OR Jack chooses voluntary strategic concession

Communication:
  - 5 message types (see MESSAGE_TYPES)
  - Players can lie — broadcast any symbol regardless of truth
  - Messages are public (all players observe all messages)

Observation asymmetry:
  - Jack: observes full suspicion distribution of all players
  - Regular players: observe aggregate suspicion signal only
  - 1-2 "perceptive" regular players: partial suspicion visibility

Win conditions:
  - Jack: n_alive_regular == 0
  - Regular: jack_suspicion_consensus == 1.0 (all surviving
    players certain) → forced concession
  - Jack voluntary concession: Jack chooses concede action
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import math


# ── Constants ──────────────────────────────────────────────────────────────

N_PLAYERS       = 20
N_REGULAR       = 19
N_SUITS         = 4          # Hearts, Diamonds, Clubs, Spades
N_MESSAGES      = 5
SUSPICION_THRESHOLD = 0.85   # above this = highly suspicious
CONSENSUS_THRESHOLD = 1.0    # 100% of survivors must be certain

# Message types
# 0: broadcast a symbol (can lie — agent chooses which symbol)
# 1: accuse a specific player of being the Jack
# 2: defend a specific player (claim they are not Jack)
# 3: deny being the Jack
# 4: silence

MESSAGE_TYPES = {
    0: "broadcast_symbol",
    1: "accuse",
    2: "defend",
    3: "deny",
    4: "silence",
}

SUITS = {0: "Hearts", 1: "Diamonds", 2: "Clubs", 3: "Spades"}

# Reward structure
JACK_WIN_REWARD     = +10.0   # Jack wipes lobby
JACK_LOSE_REWARD    = -10.0   # Jack forced to concede
JACK_ELIM_REWARD    = +0.5    # per regular player eliminated
REGULAR_WIN_REWARD  = +10.0   # regular players corner Jack
REGULAR_LOSE_REWARD = -10.0   # regular player eliminated
REGULAR_SURV_REWARD = +0.1    # per round survived


# ── Game State ─────────────────────────────────────────────────────────────

class PlayerState:
    """Tracks one player's state throughout the game."""

    def __init__(self, player_id: int, is_jack: bool,
                 true_symbol: int, n_perceptive: int = 2):
        self.player_id   = player_id
        self.is_jack     = is_jack
        self.true_symbol = true_symbol
        self.alive       = True
        self.is_perceptive = (
            not is_jack and player_id < n_perceptive
        )

        # What this player believes about who is the Jack
        # suspicion[i] = probability player i is the Jack
        # initialized uniform over all non-self players
        self.suspicion = np.zeros(N_PLAYERS, dtype=np.float32)

        # What symbol this player thinks they have
        # Regular players: uniform prior (don't know own symbol)
        # Jack: knows exactly (prosthetic mirror)
        if is_jack:
            self.own_symbol_belief = np.zeros(N_SUITS, dtype=np.float32)
            self.own_symbol_belief[true_symbol] = 1.0
        else:
            self.own_symbol_belief = np.ones(
                N_SUITS, dtype=np.float32) / N_SUITS

    def update_suspicion(self, target: int, delta: float):
        """Adjust suspicion toward target player, renormalize."""
        self.suspicion[target] = np.clip(
            self.suspicion[target] + delta, 0.0, 1.0)
        total = self.suspicion.sum()
        if total > 0:
            self.suspicion /= total

    def update_own_belief(self, symbol: int, confidence: float):
        """Update belief about own symbol from observed messages."""
        self.own_symbol_belief[symbol] += confidence
        total = self.own_symbol_belief.sum()
        if total > 0:
            self.own_symbol_belief /= total


# ── Jack of Hearts Environment ─────────────────────────────────────────────

class JackOfHeartsEnv(gym.Env):
    """
    Multi-agent Jack of Hearts environment.

    This environment models ONE agent's perspective at a time.
    For full multi-agent training, instantiate N_PLAYERS copies
    sharing game state via a GameCoordinator (built separately).

    For initial experiments, we run the Jack agent against
    rule-based regular players to establish baseline behavior,
    then upgrade to full PPO regular agents.

    Parameters
    ----------
    agent_role : str
        'jack' or 'regular' — which agent this env controls
    agent_id : int
        Player index (0 = Jack by convention)
    n_perceptive : int
        Number of regular players with partial suspicion visibility
    seed : int
        Random seed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_role: str = "jack",
        agent_id: int = 0,
        n_perceptive: int = 2,
        seed: int = 42,
    ):
        super().__init__()

        assert agent_role in ("jack", "regular")
        self.agent_role   = agent_role
        self.agent_id     = agent_id
        self.n_perceptive = n_perceptive
        self.rng          = np.random.default_rng(seed)

        # ── Observation space ──────────────────────────────────
        # Shared components (all agents):
        #   - round number (normalized)                    1
        #   - n_alive (normalized)                         1
        #   - own symbol belief                            4
        #   - last round's public messages (N_PLAYERS x 2) 40
        #     (message_type, target_or_symbol)
        #   - alive mask                                   20
        # Jack-only additional:
        #   - full suspicion distribution                  20
        # Regular (perceptive) additional:
        #   - aggregate suspicion signal                   20
        # Regular (non-perceptive):
        #   - own suspicion level only                     1

        if agent_role == "jack":
            obs_dim = 1 + 1 + N_SUITS + (N_PLAYERS * 2) + N_PLAYERS + N_PLAYERS
        elif n_perceptive > 0:
            obs_dim = 1 + 1 + N_SUITS + (N_PLAYERS * 2) + N_PLAYERS + N_PLAYERS
        else:
            obs_dim = 1 + 1 + N_SUITS + (N_PLAYERS * 2) + N_PLAYERS + 1

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ───────────────────────────────────────
        # Each step = one round action:
        #   [0]: message_type (0-4, discretized from continuous)
        #   [1]: target player or symbol (0-19 for player,
        #        0-3 for symbol, context-dependent)
        #   [2]: declaration symbol (0-3, what to declare
        #        as own symbol this round)
        #   [3]: concede (Jack only, >0.5 = concede)

        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32
        )

        # Game state (initialized in reset)
        self.players      : list[PlayerState] = []
        self.round        : int = 0
        self.game_over    : bool = False
        self.winner       : Optional[str] = None
        self.symbols      : np.ndarray = np.array([])
        self.alive_mask   : np.ndarray = np.array([])
        self.last_messages: list = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Assign symbols randomly
        self.symbols = self.rng.integers(
            0, N_SUITS, size=N_PLAYERS).astype(np.int32)

        # Initialize player states
        self.players = []
        for i in range(N_PLAYERS):
            is_jack = (i == 0)  # player 0 is always Jack
            self.players.append(PlayerState(
                player_id=i,
                is_jack=is_jack,
                true_symbol=int(self.symbols[i]),
                n_perceptive=self.n_perceptive,
            ))

        # Initialize suspicion uniform over others
        for p in self.players:
            for i in range(N_PLAYERS):
                if i != p.player_id:
                    p.suspicion[i] = 1.0 / (N_PLAYERS - 1)

        self.round        = 0
        self.game_over    = False
        self.winner       = None
        self.alive_mask   = np.ones(N_PLAYERS, dtype=np.float32)
        self.last_messages = [(4, 0)] * N_PLAYERS  # all silent

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for this agent."""
        agent = self.players[self.agent_id]

        # Round progress and population
        round_norm  = min(self.round / 50.0, 1.0)
        alive_norm  = self.alive_mask.sum() / N_PLAYERS

        # Own symbol belief
        own_belief = agent.own_symbol_belief.copy()

        # Last round messages: (message_type/4, target/19)
        msg_flat = np.zeros(N_PLAYERS * 2, dtype=np.float32)
        for i, (mtype, mtarget) in enumerate(self.last_messages):
            msg_flat[i * 2]     = mtype / (N_MESSAGES - 1)
            msg_flat[i * 2 + 1] = mtarget / max(N_PLAYERS - 1, 1)

        # Alive mask
        alive = self.alive_mask.copy()

        # Suspicion component
        if self.agent_role == "jack":
            # Full suspicion of all players toward agent
            suspicion_obs = np.array([
                p.suspicion[self.agent_id]
                for p in self.players
            ], dtype=np.float32)
        elif agent.is_perceptive:
            # Partial: observe all players' top suspicion target
            suspicion_obs = np.array([
                p.suspicion.max() for p in self.players
            ], dtype=np.float32)
        else:
            # Only own suspicion level (how suspected am I?)
            own_suspicion = np.mean([
                p.suspicion[self.agent_id]
                for p in self.players
                if p.player_id != self.agent_id
            ])
            suspicion_obs = np.array([own_suspicion], dtype=np.float32)

        obs = np.concatenate([
            [round_norm, alive_norm],
            own_belief,
            msg_flat,
            alive,
            suspicion_obs,
        ]).astype(np.float32)

        # Clip to observation space bounds
        return np.clip(obs, -1.0, 1.0)

    def step(self, action: np.ndarray):
        """
        Process one round of the game for this agent.
        Full multi-agent coordination handled by GameCoordinator.
        For now returns single-agent step for architecture testing.
        """
        # Decode action
        msg_type   = int(action[0] * (N_MESSAGES - 1) + 0.5)
        msg_type   = np.clip(msg_type, 0, N_MESSAGES - 1)
        msg_target = int(action[1] * (N_PLAYERS - 1) + 0.5)
        msg_target = np.clip(msg_target, 0, N_PLAYERS - 1)
        declaration = int(action[2] * (N_SUITS - 1) + 0.5)
        # Jack always knows his symbol — override random declaration
        if self.agent_role == "jack":
            declaration = int(self.players[self.agent_id].true_symbol)
        declaration = np.clip(declaration, 0, N_SUITS - 1)
        concede    = float(action[3]) > 0.85

        agent = self.players[self.agent_id]

        # Jack voluntary concession
        if self.agent_role == "jack" and concede:
            self.game_over = True
            self.winner    = "regular"
            reward = JACK_LOSE_REWARD
            return self._get_obs(), reward, True, False, {
                "outcome": "jack_voluntary_concession",
                "round": self.round,
            }

        # Process declaration
        true_sym = agent.true_symbol
        survived = (declaration == true_sym)

        if not survived and agent.alive:
            agent.alive       = False
            self.alive_mask[self.agent_id] = 0.0

        # Update messages
        self.last_messages[self.agent_id] = (msg_type, msg_target)

        # Update suspicion based on messages
        self._update_suspicion_from_message(
            self.agent_id, msg_type, msg_target)

        # Check win conditions
        reward = self._check_win_conditions(survived)

        self.round += 1

        return self._get_obs(), reward, self.game_over, False, {
            "round":       self.round,
            "alive":       int(self.alive_mask.sum()),
            "survived":    survived,
            "declaration": declaration,
            "true_symbol": true_sym,
            "outcome":     self.winner or "ongoing",
        }

    def _update_suspicion_from_message(
        self, sender: int, msg_type: int, target: int
    ):
        """Update all players' suspicion based on observed message."""
        for p in self.players:
            if not p.alive or p.player_id == sender:
                continue
            if msg_type == 1:   # accusation
                p.update_suspicion(target, +0.1)
            elif msg_type == 2: # defense
                p.update_suspicion(target, -0.05)
            elif msg_type == 3: # denial by sender
                p.update_suspicion(sender, -0.05)

    def _check_win_conditions(self, agent_survived: bool) -> float:
        """Check win conditions and return reward."""
        n_alive_regular = sum(
            1 for p in self.players
            if p.alive and not p.is_jack
        )

        # Jack wins: all regular players dead
        if n_alive_regular == 0:
            self.game_over = True
            self.winner    = "jack"
            return JACK_WIN_REWARD if self.agent_role == "jack" \
                   else REGULAR_LOSE_REWARD

        # Regular players win: 100% suspicion consensus on Jack
        jack_suspicion = np.array([
            p.suspicion[0]  # player 0 is Jack
            for p in self.players
            if p.alive and not p.is_jack
        ])
        if len(jack_suspicion) > 0 and \
           (jack_suspicion >= SUSPICION_THRESHOLD).all():
            self.game_over = True
            self.winner    = "regular"
            return REGULAR_WIN_REWARD if self.agent_role == "regular" \
                   else JACK_LOSE_REWARD

        # Game ongoing
        if self.agent_role == "regular":
            return REGULAR_SURV_REWARD if agent_survived \
                   else REGULAR_LOSE_REWARD
        else:
            return JACK_ELIM_REWARD * (1 - agent_survived)

    def render(self):
        pass