# Research Log

## Session 001 — 2026-05-31

**What I tried:**
Project initialization. Set up repository structure, installed 
Python 3.11.4 dependencies (numpy, pandas, matplotlib, plotly, 
stable-baselines3, gymnasium, scipy, torch).

**What happened:**
Environment fully configured. All packages installed without 
conflicts. VS Code connected to local repo at 
C:\Users\anisz\bounded-rationality-games.

**Hypothesis for why:**
N/A — setup session.

**Next step:**
Write p-beauty contest environment (environments/p_beauty_contest.py).
5 players, p=2/3. Start with winner-take-all reward to establish 
baseline behavior before testing proximity reward.

## Session 002 — 2026-05-31

**What I tried:**
Implemented PBeautyContest environment (environments/p_beauty_contest.py).
5 players, p=2/3, winner_take_all and proximity reward types.
Mixed opponent population: 40% L1, 35% L2, 15% L0, 10% L3+
based on Nagel (1995) empirical distribution.
Ran sanity check (test_env.py).

**What happened:**
All 5 tests passed cleanly.
- Observation shape: (4,) ✓
- Level-k values verified (L0=50, L1=33.3, L2=22.2) ✓
- Implied level formula accurate to 2 decimal places ✓
- Full episode runs 100 rounds without errors ✓
- Random agent mean submission: 52.737, implied level: 1.115

**Hypothesis for why:**
N/A — clean first implementation.

**Next step:**
Train PPO agent with winner_take_all reward first (expected
failure — reward too sparse). Document the failure, then
switch to proximity reward.

## Session 003 — 2026-05-31

**What I tried:**
EXP001: winner_take_all reward, 50k timesteps.
EXP002: proximity reward, 200k timesteps.
Both: 5 players, p=2/3, mixed opponent population.

**What happened:**
EXP001: Agent did NOT produce flat curve as predicted.
Instead converged to mean submission ~7.1, implied level 4.8.
Overshooting toward Nash rather than finding human equilibrium.
Failure mode: wrong direction, not no learning.

EXP002: Clean convergence. Stabilized at implied level 2.726
after ~150k steps. Nagel human benchmark: 1.8.
Gap of 0.926 levels above human mean.

**Hypothesis for why:**
EXP001: Winner-take-all with mixed opponents rewards
submitting just below the opponent cluster. Agent learned
to undercut opponents rather than model human reasoning depth.
Correct optimization, wrong objective.

EXP002: Proximity reward produces correct learning direction.
Gap vs Nagel (2.726 vs 1.8) likely reflects training regime
difference — RL agent has 200k rounds of feedback vs human
subjects who played once. Extensive experience produces
slightly deeper reasoning than one-shot human play.

**Next step:**
Run EXP002 longer (500k steps) to see if level continues
descending toward 1.8 or has genuinely stabilized at 2.7.
Also need to test whether the gap closes when opponent
population is fixed at level-1 only (simpler target).

## Session 004 — 2026-05-31

**What I tried:**
EXP003: proximity reward, fixed Level-1 opponents, 200k steps.

**What happened:**
Converged to implied level 2.201 (vs 2.726 with mixed opponents).
Gap to Nagel partially closed: 0.926 → 0.401.
Near-perfect reward (99.97/100). Clean plateau from step 120k.

**Hypothesis for why:**
Gap decomposition:
- Population effect: ~0.5 levels (mixed vs fixed L1 opponents)
- Training regime effect: ~0.4 levels (200k rounds vs one-shot
  human experiment)
Total gap to Nagel: 0.401 levels with simplest opponent type.

**Next step:**
Design EXP004 — vary p value (try p=0.5 and p=0.9) to test
whether reasoning depth scales with p as theory predicts.
Or move to multi-agent version where agents play against
each other rather than fixed opponents.

## Session 005 — 2026-05-31

**What I tried:**
EXP004: proximity reward, p=0.5, mixed opponents, 200k steps.
EXP005: proximity reward, p=0.9, mixed opponents, 200k steps.
Testing whether implied reasoning depth scales with p as
level-k theory predicts.

**What happened:**
EXP004 (p=0.5): Converged to implied level 2.530.
Almost identical to EXP002 (p=2/3, level 2.726).
Raw submission dropped from 16.6 to 8.7 — correct direction.
Implied level roughly invariant to p — consistent with theory.

EXP005 (p=0.9): Did NOT converge. Stuck at level 10 until
step 170k, then slowly descending. Final level 8.297 at 200k,
curve still declining. Model moved to experiments/failed/.

**Hypothesis for why:**
EXP004: Level-k reasoning depth is a property of the agent,
not the game parameter. Agent adjusts submission proportionally
to p while maintaining same reasoning depth. Confirms agent
has learned the underlying level-k structure.

EXP005: With p=0.9, level-k submissions cluster near 50
(L1=45, L2=40.5, L3=36.5). Proximity reward gradient is
nearly flat across this region — weak learning signal.
Convergence time scales with p. Need 500k steps minimum.

**Next step:**
EXP005b: Run p=0.9 to 500k steps to find true plateau.
Also move EXP005 to experiments/failed/ as documented
non-convergence.

## Session 006 — 2026-05-31

**What I tried:**
EXP005b: p=0.9, proximity reward, 500k steps.
Extended from EXP005 which had not converged at 200k.

**What happened:**
Converged to implied level 2.814 at approximately step 330k.
Final mean submission 37.172 — correctly scaled for p=0.9.
Plateau confirmed from step 330k through 500k.

**Key finding — p-invariance confirmed:**
p=0.5 → 2.530, p=2/3 → 2.726, p=0.9 → 2.814.
Implied level approximately invariant to p.
Consistent with level-k theory prediction.

**Secondary finding — convergence scaling:**
Convergence steps: p=0.5 (~100k), p=2/3 (~150k), p=0.9 (~330k).
Higher p = flatter reward landscape = slower convergence.

**Beauty contest experiment battery complete.**
All five experiments done. Three publishable findings.

**Next session (June 1):**
Multi-agent beauty contest design — agents play
against each other rather than fixed opponents.

## Session 007 — 2026-06-01

**What I tried:**
EXP006: Multi-agent beauty contest, parameter sharing PPO.
All 5 agents share one live policy (no deepcopy — true parameter sharing).
p=2/3, 200k steps, seed=42. Comparison target: EXP002 level 2.726.

**What happened:**
Policy converged to near-Nash equilibrium. Final mean submission 0.284,
implied level 10.000 (capped). Deterministic evaluation collapses to ~0.

**Logging bug noted:**
_last_mean reads 50.00 throughout training due to reading post-reset value.
Actual within-episode submissions are lower. Result is valid — deterministic
evaluation is authoritative. Fix logging in next session.

**Key finding — Nash-seeking via parameter sharing:**
Parameter sharing creates a symmetric downward gradient. When all agents
share one policy, changing the policy changes ALL agents' submissions
simultaneously. Target = p × mean follows submissions downward. No gradient
signal stops convergence before zero. Symmetric Nash equilibrium (submit 0)
is the fixed point.

This is a different mechanism from EXP001 (winner-take-all reward).
EXP001: wrong reward → Nash-seeking.
EXP006: right reward + parameter sharing → Nash-seeking.

**2x2 architecture table now complete:**

| | Fixed opponents | Shared policy |
|---|---|---|
| Winner-take-all | Nash-seeking (EXP001) | [untested] |
| Proximity | Bounded rationality (EXP002-005) | Nash-seeking (EXP006) |

**Finding:** Bounded rationality requires BOTH proximity reward AND fixed
opponent distribution. Changing either condition produces Nash-seeking.
Opponent structure is a joint determinant alongside reward architecture.

**Next session:**
Fix _last_mean logging bug. Design EXP007: independent PPO agents
(no parameter sharing) with proximity reward. Tests whether co-evolution
without shared gradients also produces Nash or recovers bounded rationality.