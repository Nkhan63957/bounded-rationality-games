"""
Rule-Based Regular Player
===========================
Intelligent rule-based agent for regular players.
Replaces random actions with sensible heuristics so
games last multiple rounds and produce meaningful data.

Strategy phases:
  Phase 1 (rounds 1-2): Cooperative
    - Offer alliance to nearest neighbor
    - Tell alliance partner what symbol you see on their collar
    - Declare symbol you are LEAST likely to be

  Phase 2 (rounds 3+): Strategic  
    - Cross-reference symbol claims from multiple sources
    - Raise suspicion on players with inconsistent claims
    - Accuse highest-suspicion player
    - Declare most confident own symbol
"""

import numpy as np
from environments.jack_of_hearts import (
    N_PLAYERS, N_SUITS, N_MESSAGES, NEIGHBORHOOD_SIZE
)


class RuleBasedRegular:
    """
    Rule-based regular player agent.

    Parameters
    ----------
    player_id : int
        This agent's player index
    neighborhood : list
        Player ids this agent can see
    seed : int
    """

    def __init__(self, player_id: int,
                 neighborhood: list, seed: int = 42):
        self.player_id    = player_id
        self.neighborhood = neighborhood
        self.rng          = np.random.default_rng(seed)
        self.round        = 0

        # What different sources have told me about my own symbol
        # {informer_id: claimed_symbol}
        self.received_claims: dict = {}

        # My current best guess at own symbol
        self.own_symbol_estimate = np.ones(
            N_SUITS, dtype=np.float32) / N_SUITS

        # Alliance state
        self.current_ally: int = -1  # -1 = no ally

    def act(self, obs: np.ndarray, env_state: dict) -> np.ndarray:
        """
        Produce an 8-dimensional action vector.

        Parameters
        ----------
        obs : np.ndarray shape (146,)
        env_state : dict with keys:
            'alive_mask'     : np.ndarray shape (20,)
            'symbols'        : np.ndarray shape (20,) — visible symbols
            'player_states'  : list of PlayerState
            'round'          : int

        Returns
        -------
        action : np.ndarray shape (8,)
        """
        action = np.zeros(8, dtype=np.float32)
        alive_mask   = env_state["alive_mask"]
        symbols      = env_state["symbols"]
        player_states = env_state["player_states"]
        self.round   = env_state["round"]

        my_state = player_states[self.player_id]

        # Update own symbol estimate from received claims
        self._update_own_estimate(my_state)

        # ── Action [2]: Declaration ────────────────────────────
        # Declare the symbol we're most confident we ARE
        best_sym = int(np.argmax(self.own_symbol_estimate))
        action[2] = best_sym / (N_SUITS - 1)

        # ── Alliance logic ─────────────────────────────────────
        alive_neighbors = [
            n for n in self.neighborhood
            if alive_mask[n] > 0 and n != self.player_id
        ]

        if self.round <= 2:
            # Phase 1: seek alliance with first alive neighbor
            if self.current_ally == -1 and alive_neighbors:
                self.current_ally = alive_neighbors[0]

            if self.current_ally != -1 and \
               alive_mask[self.current_ally] > 0:
                # Offer/renew alliance
                action[4] = 1.0  # alliance_action = offer
                action[5] = self.current_ally / (N_PLAYERS - 1)

                # Tell ally what symbol we see on their collar
                if self.current_ally in self.neighborhood:
                    ally_true_sym = int(symbols[self.current_ally])
                    # Can lie — for rule-based, be honest
                    action[0] = 0.0 / (N_MESSAGES - 1)  # msg_type 0
                    action[1] = self.current_ally / (N_PLAYERS - 1)
                    action[3] = ally_true_sym / (N_SUITS - 1)
            else:
                action[4] = 0.0  # no alliance offer

        else:
            # Phase 2: strategic
            # Renew alliance if ally still alive
            if self.current_ally != -1 and \
               alive_mask[self.current_ally] > 0:
                action[4] = 1.0
                action[5] = self.current_ally / (N_PLAYERS - 1)

            # Find most suspicious player
            suspicion = my_state.suspicion.copy()
            suspicion[self.player_id] = 0.0  # can't accuse self
            # Zero out dead players
            for i in range(N_PLAYERS):
                if alive_mask[i] == 0:
                    suspicion[i] = 0.0

            most_suspicious = int(np.argmax(suspicion))

            if suspicion[most_suspicious] > 0.3:
                # Accuse most suspicious player
                action[0] = 1.0 / (N_MESSAGES - 1)  # msg_type 1
                action[1] = most_suspicious / (N_PLAYERS - 1)
            elif alive_neighbors:
                # Share symbol info with a neighbor
                target = self.rng.choice(alive_neighbors)
                if target in self.neighborhood:
                    target_sym = int(symbols[target])
                    action[0] = 0.0 / (N_MESSAGES - 1)
                    action[1] = target / (N_PLAYERS - 1)
                    action[3] = target_sym / (N_SUITS - 1)

        # [6] concede = 0 (regular players don't concede)
        # [7] manipulate = 0 (regular players can't manipulate)
        action[6] = 0.0
        action[7] = 0.0

        return action

    def _update_own_estimate(self, my_state):
        """
        Update own symbol estimate from:
        1. Alliance claims (what allies said I'm wearing)
        2. Neighborhood inference (symbol distribution)
        """
        # From alliance claims
        if my_state.alliance_claims:
            claim_votes = np.zeros(N_SUITS, dtype=np.float32)
            for informer_id, sym in my_state.alliance_claims.items():
                claim_votes[sym] += 1.0
            if claim_votes.sum() > 0:
                claim_dist = claim_votes / claim_votes.sum()
                # Blend: 60% claims, 40% prior
                self.own_symbol_estimate = (
                    0.6 * claim_dist +
                    0.4 * my_state.own_symbol_belief
                )
                self.own_symbol_estimate /= \
                    self.own_symbol_estimate.sum()
        else:
            self.own_symbol_estimate = my_state.own_symbol_belief.copy()