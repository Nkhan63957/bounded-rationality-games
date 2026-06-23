"""
trade_game.py  —  Two-country tariff game (MAKOTO strategic-interaction core)

A reduced-form trade-policy game grounded in optimal-tariff theory
(Johnson 1953-54, "Optimal Tariffs and Retaliation"; Grossman-Helpman 1995;
Ossa 2014, "Trade Wars and Trade Talks with Data"). Replaces the inert
additive fiscal spillover with a genuine strategic-interaction channel:
each country's best response depends on the partner's action, the
non-cooperative equilibrium is Pareto-dominated, and the cooperative
optimum is free trade. This is the analytic benchmark against which the
boundedly-rational PPO learners are evaluated, on neutral welfare units.

Welfare for country i (deviation from free-trade welfare, model units):

    W_i(t_i, t_j) =  beta * t_i          # terms-of-trade gain: tariff shifts
                                          #   world prices in i's favor, a
                                          #   *transfer* extracted from j
                   - beta * t_j           # ToT loss: j extracts from i
                                          #   (zero-sum, cancels in joint W)
                   - 0.5 * kappa * t_i**2 # own deadweight loss (convex)
                   - delta * t_i * t_j    # bilateral volume contraction
                                          #   (the more both tax, the less
                                          #    trade remains; hurts both)

Because the ToT terms are pure transfers, the joint objective collapses to
  W_i + W_j = -0.5*kappa*(t_i^2 + t_j^2) - 2*delta*t_i*t_j
which is maximized at free trade (0, 0) -> the cooperative optimum.

Calibration note (researcher to anchor): the ratio beta/kappa equals the
unconstrained optimal tariff, which optimal-tariff theory ties to the inverse
foreign export-supply elasticity (t_opt ~ 1/eps_x). Choose beta, kappa from
estimated US-China bilateral trade elasticities; choose delta from the
bilateral trade-volume response to tariffs. Defaults below put the Nash
tariff near ~38% and a clear coordination gap, in the range of large-country
estimates in Ossa (2014).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class TradeWarGame:
    beta: float = 0.50     # terms-of-trade extraction strength
    kappa: float = 1.00    # own deadweight-loss curvature
    delta: float = 0.30    # bilateral trade-contraction (strategic) coupling
    tau_max: float = 1.00  # prohibitive-tariff cap (tariffs are non-negative)

    # ---- payoff ----
    def welfare(self, t_i: float, t_j: float) -> float:
        return (self.beta * t_i
                - self.beta * t_j
                - 0.5 * self.kappa * t_i ** 2
                - self.delta * t_i * t_j)

    # ---- single-country best response (closed form, clipped to [0, tau_max]) ----
    def best_response(self, t_j: float) -> float:
        # argmax_t_i  beta*t_i - 0.5*kappa*t_i^2 - delta*t_i*t_j
        # FOC: beta - kappa*t_i - delta*t_j = 0
        t_star = (self.beta - self.delta * t_j) / self.kappa
        return float(np.clip(t_star, 0.0, self.tau_max))

    # ---- non-cooperative Nash via best-response iteration ----
    def nash(self, iters: int = 200, tol: float = 1e-12) -> tuple[float, float]:
        ti = tj = 0.0
        for _ in range(iters):
            ni, nj = self.best_response(tj), self.best_response(ti)
            if abs(ni - ti) < tol and abs(nj - tj) < tol:
                ti, tj = ni, nj
                break
            ti, tj = ni, nj
        return ti, tj

    # ---- cooperative optimum: maximize joint welfare (solved numerically) ----
    def cooperative(self, n: int = 2001) -> tuple[float, float]:
        grid = np.linspace(0.0, self.tau_max, n)
        # joint welfare is symmetric & separable-after-transfer; search symmetric
        # diagonal first, then confirm no asymmetric improvement on a coarse grid
        Ti, Tj = np.meshgrid(grid, grid, indexing="ij")
        joint = (-0.5 * self.kappa * (Ti ** 2 + Tj ** 2) - 2 * self.delta * Ti * Tj)
        k = np.unravel_index(np.argmax(joint), joint.shape)
        return float(grid[k[0]]), float(grid[k[1]])

    # ---- diagnostics ----
    def coordination_gap(self) -> dict:
        ni, nj = self.nash()
        ci, cj = self.cooperative()
        W_nash = self.welfare(ni, nj) + self.welfare(nj, ni)
        W_coop = self.welfare(ci, cj) + self.welfare(cj, ci)
        # price of anarchy on the welfare *loss* relative to free trade
        return dict(nash=(ni, nj), coop=(ci, cj),
                    W_nash_joint=W_nash, W_coop_joint=W_coop,
                    gap=W_coop - W_nash)

    def is_prisoners_dilemma(self) -> dict:
        """Verify the three PD properties numerically."""
        ci, cj = self.cooperative()
        ni, nj = self.nash()
        # (1) unilateral defection from cooperation is individually rational
        br_from_coop = self.best_response(cj)
        defect_gain = (self.welfare(br_from_coop, cj) - self.welfare(ci, cj))
        # (2) Nash is Pareto-dominated by cooperation (per-country)
        pareto_dom = (self.welfare(ci, cj) > self.welfare(ni, nj))
        # (3) best response genuinely depends on partner (strategic interaction)
        slope = self.best_response(0.0) - self.best_response(self.tau_max)
        return dict(defect_gain=defect_gain,
                    defection_profitable=defect_gain > 1e-9,
                    nash_pareto_dominated=pareto_dom,
                    best_response_slope=-(slope / self.tau_max),
                    strategic_interaction=abs(slope) > 1e-9)

    # ---- RL bridge: continuous action a in [-1,1] -> tariff in [0, tau_max] ----
    def action_to_tariff(self, a: float) -> float:
        return float(np.clip(0.5 * (a + 1.0) * self.tau_max, 0.0, self.tau_max))


if __name__ == "__main__":
    g = TradeWarGame()
    cg = g.coordination_gap()
    pd = g.is_prisoners_dilemma()
    print("=" * 58)
    print("TRADE-WAR STAGE GAME  —  validation report")
    print("=" * 58)
    print(f"params: beta={g.beta}  kappa={g.kappa}  delta={g.delta}")
    print(f"unconstrained optimal tariff (beta/kappa): {g.beta/g.kappa:.3f}")
    print("-" * 58)
    print(f"Cooperative optimum (free trade) : {cg['coop']}")
    print(f"Non-cooperative Nash             : ({cg['nash'][0]:.4f}, {cg['nash'][1]:.4f})")
    print(f"  closed-form Nash beta/(kappa+delta): {g.beta/(g.kappa+g.delta):.4f}")
    print(f"Joint welfare  @ cooperation     : {cg['W_coop_joint']:+.4f}")
    print(f"Joint welfare  @ Nash            : {cg['W_nash_joint']:+.4f}")
    print(f"Coordination gap (welfare units) : {cg['gap']:+.4f}")
    print("-" * 58)
    print("Prisoner's-dilemma checks:")
    print(f"  (1) defection from cooperation profitable : {pd['defection_profitable']}  (gain {pd['defect_gain']:+.4f})")
    print(f"  (2) Nash Pareto-dominated by cooperation  : {pd['nash_pareto_dominated']}")
    print(f"  (3) genuine strategic interaction         : {pd['strategic_interaction']}  (reaction slope {pd['best_response_slope']:+.3f})")
    print("=" * 58)
    verdict = (pd['defection_profitable'] and pd['nash_pareto_dominated']
               and pd['strategic_interaction'])
    print(f"VERDICT: {'genuine prisoner-dilemma trade game ✓' if verdict else 'NOT a valid PD ✗'}")
