"""
Jack of Hearts — Solitary Confinement Environment
===================================================
Based on Alice in Borderland Season 2, Episode 5.

Game structure:
  - 20 players: 1 Jack, 19 regular players
  - Each player wears a collar displaying one of 4 suit symbols
    visible to NEARBY players only (neighborhood-based visibility)
  - Jack has a hidden prosthetic mirror — he can see his own symbol
  - Each round: alliance phase → communication → declaration
  - Wrong or missing declaration → collar explosion → elimination
  - Jack wins: entire lobby wiped (all regular players dead)
  - Regular players win: 100% suspicion consensus → forced concession
    OR Jack chooses voluntary strategic concession

Alliance mechanics:
  - Players autonomously form/break alliances each round
  - Within alliance: symbol claims shared (but can lie)
  - Defection: remove from alliance, suspicion of ex-partner rises
  - Alliance offers are mutual — both must offer to form

Deception:
  - ALL players can lie to ANYONE including alliance partners
  - Jack has additional manipulation action (diminishing returns)
  - Manipulation reduces target's suspicion of Jack

Action space (8-dimensional, uniform for all players):
  [0]: message_type     — 0-4 (see MESSAGE_TYPES)
  [1]: message_target   — which player to direct message at
  [2]: declaration      — which symbol to declare as own
  [3]: broadcast_symbol — which symbol to claim in message (can lie)
  [4]: alliance_action  — >0.5 = offer/renew alliance with target
  [5]: alliance_target  — which player to offer alliance to
  [6]: concede          — Jack only, >0.85 = voluntary concession
  [7]: manipulate       — Jack only, which player to brainwash

Observation space additions vs v1:
  - alliance_mask       — who am I currently allied with (20-dim)
  - alliance_claims     — what allies claimed their symbol is (20-dim)
  - manipulation_eff    — Jack only, remaining effectiveness (20-dim)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

N_PLAYERS           = 20
N_REGULAR           = 19
N_SUITS             = 4
N_MESSAGES          = 5
SUSPICION_THRESHOLD = 0.85
CONSENSUS_THRESHOLD = 1.0
NEIGHBORHOOD_SIZE   = 6    # players visible to each player

# Manipulation mechanics
MANIP_BASE_EFFECT   = 0.30  # first use drops suspicion by 0.30
MANIP_DECAY         = 0.60  # each repeat multiplied by 0.60

# Message types
MESSAGE_TYPES = {
    0: "broadcast_symbol",   # claim a symbol (can lie)
    1: "accuse",             # accuse specific player of being Jack
    2: "defend",             # claim specific player is not Jack
    3: "deny",               # deny being the Jack
    4: "silence",            # say nothing
}

SUITS = {0: "Hearts", 1: "Diamonds", 2: "Clubs", 3: "Spades"}

# Rewards
JACK_WIN_REWARD     = +10.0
JACK_LOSE_REWARD    = -10.0
JACK_ELIM_REWARD    = +0.5
REGULAR_WIN_REWARD  = +10.0
REGULAR_LOSE_REWARD = -10.0
REGULAR_SURV_REWARD = +0.1
ALLIANCE_SURV_BONUS = +0.05  # bonus for each round an alliance partner survives


# ── Player State ───────────────────────────────────────────────────────────

class PlayerState:
    """Tracks one player's state throughout the game."""

    def __init__(self, player_id: int, is_jack: bool,
                 true_symbol: int, neighborhood: list,
                 n_perceptive: int = 2):
        self.player_id    = player_id
        self.is_jack      = is_jack
        self.true_symbol  = true_symbol
        self.alive        = True
        self.neighborhood = neighborhood  # list of visible player ids
        self.is_perceptive = (not is_jack and player_id < n_perceptive)

        # Alliance tracking
        self.alliances: set = set()          # current allied player ids
        self.pending_offer: Optional[int] = None  # outgoing alliance offer
        self.alliance_claims: dict = {}      # {player_id: claimed_symbol}
        self.manipulation_counts: dict = {}  # Jack only: {target: times}

        # Beliefs
        self.suspicion = np.zeros(N_PLAYERS, dtype=np.float32)

        # Own symbol belief
        if is_jack:
            self.own_symbol_belief = np.zeros(N_SUITS, dtype=np.float32)
            self.own_symbol_belief[true_symbol] = 1.0
        else:
            self.own_symbol_belief = np.ones(
                N_SUITS, dtype=np.float32) / N_SUITS

    def update_suspicion(self, target: int, delta: float):
        self.suspicion[target] = np.clip(
            self.suspicion[target] + delta, 0.0, 1.0)
        total = self.suspicion.sum()
        if total > 0:
            self.suspicion /= total

    def update_own_belief(self, symbol: int, confidence: float):
        self.own_symbol_belief[symbol] += confidence
        total = self.own_symbol_belief.sum()
        if total > 0:
            self.own_symbol_belief /= total

    def infer_own_symbol(self):
        """
        Infer own symbol from visible neighbors' collars.
        Symbol distribution observed → own symbol is likely
        underrepresented in neighborhood.
        """
        return self.own_symbol_belief


# ── Jack of Hearts Environment ─────────────────────────────────────────────

class JackOfHeartsEnv(gym.Env):
    """
    Single-agent perspective on the Jack of Hearts game.
    Full multi-agent coordination handled by GameCoordinator.

    Observation space (varies by role):
      Shared (all agents):
        round_norm          1
        alive_norm          1
        own_symbol_belief   4
        last_messages       N_PLAYERS * 2 = 40
        alive_mask          20
        alliance_mask       20   (who am I allied with)
        alliance_claims     20   (what allies claimed their symbol is,
                                  -1 if no claim, normalized /3)
      Jack additional:
        full suspicion dist 20
        manip_effectiveness 20
      Regular perceptive:
        aggregate suspicion 20
      Regular non-perceptive:
        own suspicion only   1
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
        base_dim = 1 + 1 + N_SUITS + (N_PLAYERS * 2) + N_PLAYERS + N_PLAYERS + N_PLAYERS

        if agent_role == "jack":
            # + full suspicion + manipulation effectiveness
            obs_dim = base_dim + N_PLAYERS + N_PLAYERS
        elif n_perceptive > 0:
            # + aggregate suspicion
            obs_dim = base_dim + N_PLAYERS
        else:
            # + own suspicion only
            obs_dim = base_dim + 1

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space (8-dimensional, uniform for all) ──────
        # [0] message_type     0-1 → decoded to 0-4
        # [1] message_target   0-1 → decoded to 0-19
        # [2] declaration      0-1 → decoded to 0-3
        # [3] broadcast_symbol 0-1 → decoded to 0-3 (can lie)
        # [4] alliance_action  0-1 → >0.5 = offer alliance
        # [5] alliance_target  0-1 → decoded to 0-19
        # [6] concede          0-1 → Jack only, >0.85 = concede
        # [7] manipulate       0-1 → Jack only, decoded to 0-19
        self.action_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32
        )

        # Game state
        self.players:       list  = []
        self.round:         int   = 0
        self.game_over:     bool  = False
        self.winner:        Optional[str] = None
        self.symbols:       np.ndarray = np.array([])
        self.alive_mask:    np.ndarray = np.array([])
        self.last_messages: list  = []
        self.neighborhoods: list  = []  # neighborhoods[i] = list of visible ids

        # Pending alliance offers this round: {offerer: target}
        self.pending_offers: dict = {}

    # ── Reset ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Assign symbols
        self.symbols = self.rng.integers(
            0, N_SUITS, size=N_PLAYERS).astype(np.int32)

        # Build neighborhoods (fixed ring + random cross-links)
        self.neighborhoods = self._build_neighborhoods()

        # Initialize players
        self.players = []
        for i in range(N_PLAYERS):
            is_jack = (i == 0)
            self.players.append(PlayerState(
                player_id=i,
                is_jack=is_jack,
                true_symbol=int(self.symbols[i]),
                neighborhood=self.neighborhoods[i],
                n_perceptive=self.n_perceptive,
            ))

        # Uniform suspicion initialization
        for p in self.players:
            for i in range(N_PLAYERS):
                if i != p.player_id:
                    p.suspicion[i] = 1.0 / (N_PLAYERS - 1)

        self.round        = 0
        self.game_over    = False
        self.winner       = None
        self.alive_mask   = np.ones(N_PLAYERS, dtype=np.float32)
        self.last_messages = [(4, 0)] * N_PLAYERS
        self.pending_offers = {}

        # Update own symbol beliefs from visible neighborhoods
        self._update_all_symbol_beliefs()

        return self._get_obs(), {}

    def _build_neighborhoods(self) -> list:
        """
        Build partial visibility neighborhoods.
        Each player sees NEIGHBORHOOD_SIZE others.
        Based on small-group clustering (duos, trios, groups of 6).
        """
        neighborhoods = [set() for _ in range(N_PLAYERS)]

        # Ring connectivity — everyone sees adjacent players
        for i in range(N_PLAYERS):
            for d in [-1, 1]:
                j = (i + d) % N_PLAYERS
                neighborhoods[i].add(j)
                neighborhoods[j].add(i)

        # Random additional links to reach NEIGHBORHOOD_SIZE
        for i in range(N_PLAYERS):
            while len(neighborhoods[i]) < NEIGHBORHOOD_SIZE:
                j = int(self.rng.integers(0, N_PLAYERS))
                if j != i and j not in neighborhoods[i]:
                    neighborhoods[i].add(j)
                    neighborhoods[j].add(i)

        return [list(n) for n in neighborhoods]

    def _update_all_symbol_beliefs(self):
        """
        Update each regular player's own symbol belief
        from what they can see in their neighborhood.
        """
        for p in self.players:
            if p.is_jack:
                continue
            # Count symbols visible in neighborhood
            counts = np.zeros(N_SUITS, dtype=np.float32)
            for nb_id in p.neighborhood:
                if self.players[nb_id].alive:
                    counts[self.symbols[nb_id]] += 1
            # Own symbol is likely the one least seen
            total_visible = counts.sum()
            if total_visible > 0:
                # Inverse frequency → higher belief for less-seen symbols
                inv = (total_visible - counts) + 0.1
                inv /= inv.sum()
                # Blend with prior
                p.own_symbol_belief = 0.7 * inv + 0.3 * p.own_symbol_belief
                p.own_symbol_belief /= p.own_symbol_belief.sum()

    # ── Observation ────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        agent = self.players[self.agent_id]

        round_norm = min(self.round / 50.0, 1.0)
        alive_norm = self.alive_mask.sum() / N_PLAYERS
        own_belief = agent.own_symbol_belief.copy()

        # Messages
        msg_flat = np.zeros(N_PLAYERS * 2, dtype=np.float32)
        for i, (mtype, mtarget) in enumerate(self.last_messages):
            msg_flat[i * 2]     = mtype / (N_MESSAGES - 1)
            msg_flat[i * 2 + 1] = mtarget / max(N_PLAYERS - 1, 1)

        alive = self.alive_mask.copy()

        # Alliance mask (binary: allied or not)
        alliance_mask = np.zeros(N_PLAYERS, dtype=np.float32)
        for ally_id in agent.alliances:
            alliance_mask[ally_id] = 1.0

        # Alliance claims (what allies claimed their symbol is)
        alliance_claims = np.full(N_PLAYERS, -1.0, dtype=np.float32)
        for ally_id, claimed_sym in agent.alliance_claims.items():
            alliance_claims[ally_id] = claimed_sym / (N_SUITS - 1)

        # Suspicion component
        if self.agent_role == "jack":
            suspicion_obs = np.array([
                p.suspicion[self.agent_id]
                for p in self.players
            ], dtype=np.float32)
            # Manipulation effectiveness
            manip_eff = np.zeros(N_PLAYERS, dtype=np.float32)
            for target_id, count in agent.manipulation_counts.items():
                manip_eff[target_id] = MANIP_BASE_EFFECT * (MANIP_DECAY ** count)
            extra = np.concatenate([suspicion_obs, manip_eff])

        elif agent.is_perceptive:
            suspicion_obs = np.array([
                p.suspicion.max() for p in self.players
            ], dtype=np.float32)
            extra = suspicion_obs

        else:
            own_suspicion = np.mean([
                p.suspicion[self.agent_id]
                for p in self.players
                if p.player_id != self.agent_id and p.alive
            ])
            extra = np.array([own_suspicion], dtype=np.float32)

        obs = np.concatenate([
            [round_norm, alive_norm],
            own_belief,
            msg_flat,
            alive,
            alliance_mask,
            alliance_claims,
            extra,
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── Step ───────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Single-agent step for architecture testing."""
        agent = self.players[self.agent_id]

        # Decode actions
        msg_type = int(np.clip(
            int(action[0] * (N_MESSAGES - 1) + 0.5), 0, N_MESSAGES - 1))
        msg_target = int(np.clip(
            int(action[1] * (N_PLAYERS - 1) + 0.5), 0, N_PLAYERS - 1))
        broadcast_sym = int(np.clip(
            int(action[3] * (N_SUITS - 1) + 0.5), 0, N_SUITS - 1))
        alliance_offer = float(action[4]) > 0.5
        alliance_target = int(np.clip(
            int(action[5] * (N_PLAYERS - 1) + 0.5), 0, N_PLAYERS - 1))
        concede = float(action[6]) > 0.85
        manip_target = int(np.clip(
            int(action[7] * (N_PLAYERS - 1) + 0.5), 0, N_PLAYERS - 1))

        # Jack always declares true symbol
        if self.agent_role == "jack":
            declaration = int(agent.true_symbol)
        else:
            declaration = int(np.clip(
                int(action[2] * (N_SUITS - 1) + 0.5), 0, N_SUITS - 1))

        # Jack voluntary concession
        if self.agent_role == "jack" and concede:
            self.game_over = True
            self.winner = "regular"
            return self._get_obs(), JACK_LOSE_REWARD, True, False, {
                "outcome": "jack_voluntary_concession",
                "round": self.round,
            }

        # Jack manipulation
        if self.agent_role == "jack" and manip_target != self.agent_id:
            count = agent.manipulation_counts.get(manip_target, 0)
            effect = MANIP_BASE_EFFECT * (MANIP_DECAY ** count)
            for p in self.players:
                if p.alive and p.player_id == manip_target:
                    p.update_suspicion(self.agent_id, -effect)
            agent.manipulation_counts[manip_target] = count + 1

        # Alliance offer
        if alliance_offer and alliance_target != self.agent_id:
            self.pending_offers[self.agent_id] = alliance_target

        # Message
        self.last_messages[self.agent_id] = (msg_type, msg_target)
        if msg_type == 0:  # broadcast symbol claim
            for p in self.players:
                if p.alive and p.player_id != self.agent_id:
                    if self.agent_id in p.alliances:
                        p.alliance_claims[self.agent_id] = broadcast_sym
        self._update_suspicion_from_message(
            self.agent_id, msg_type, msg_target)

        # Declaration
        survived = (declaration == agent.true_symbol)
        if not survived and agent.alive:
            agent.alive = False
            self.alive_mask[self.agent_id] = 0.0

        # Update symbol beliefs
        self._update_all_symbol_beliefs()

        reward = self._check_win_conditions(survived)
        self.round += 1

        return self._get_obs(), reward, self.game_over, False, {
            "round":       self.round,
            "alive":       int(self.alive_mask.sum()),
            "survived":    survived,
            "declaration": declaration,
            "true_symbol": int(agent.true_symbol),
            "outcome":     self.winner or "ongoing",
            "n_alliances": len(agent.alliances),
        }

    def _update_suspicion_from_message(
            self, sender: int, msg_type: int, target: int):
        for p in self.players:
            if not p.alive or p.player_id == sender:
                continue
            if msg_type == 1:
                p.update_suspicion(target, +0.10)
            elif msg_type == 2:
                p.update_suspicion(target, -0.05)
            elif msg_type == 3:
                p.update_suspicion(sender, -0.05)

    def _check_win_conditions(self, agent_survived: bool) -> float:
        n_alive_regular = sum(
            1 for p in self.players if p.alive and not p.is_jack)

        if n_alive_regular == 0:
            self.game_over = True
            self.winner = "jack"
            return JACK_WIN_REWARD if self.agent_role == "jack" \
                   else REGULAR_LOSE_REWARD

        jack_suspicion = np.array([
            p.suspicion[0]
            for p in self.players
            if p.alive and not p.is_jack
        ])
        if len(jack_suspicion) > 0 and \
                (jack_suspicion >= SUSPICION_THRESHOLD).all():
            self.game_over = True
            self.winner = "regular"
            return REGULAR_WIN_REWARD if self.agent_role == "regular" \
                   else JACK_LOSE_REWARD

        if self.agent_role == "regular":
            return REGULAR_SURV_REWARD if agent_survived \
                   else REGULAR_LOSE_REWARD
        else:
            return JACK_ELIM_REWARD * (1 - agent_survived)

    def render(self):
        pass