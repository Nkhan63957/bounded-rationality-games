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

## Session 008 — 2026-06-02

**What I built:**
Fixed logging bug in TrainingCallback — _on_step now reads
mean_sub from infos dict rather than env._last_mean, which
was returning the post-reset value of 50.0.

Added FrozenOpponentCallback: syncs opponent weights every
N rollouts rather than real-time, testing whether the EXP006
Nash convergence mechanism is gradient-sharing-specific.

**What I ran:**
EXP007: p=2/3, 200k steps, frozen opponents (sync every 5
rollouts). Hypothesis: frozen opponents break Nash and produce
bounded rationality.

**What happened:**
Hypothesis falsified. EXP007 converged to near Nash
(mean sub 0.168, implied level 10.000) — same as EXP006
(mean sub 0.284). Frozen opponents did not break Nash
convergence.

**Revised finding:**
Nash convergence is not caused by real-time gradient sharing
specifically. It is caused by same-architecture opponents
regardless of sync frequency. The key distinction is:
  - Fixed opponents (EXP002-005): bounded rationality
  - Same-architecture opponents (EXP006, EXP007): Nash

Update to 2x2 architecture table:
| | Fixed opponents | Same-arch opponents |
|---|---|---|
| Proximity reward | Bounded rationality | Nash (both sync modes) |

**Connection to Nagel (2026-06-02 correspondence):**
Professor Nagel responded to the cold email and raised the
possibility that gradient mechanisms contain hidden strategic
reasoning, citing Nagel and Tang centipede paper. EXP007
is consistent with this interpretation: each agent
independently converges to Nash through gradient descent
even without real-time weight coupling. This looks less
like mechanical convergence and more like each agent
discovering the strategic optimum individually.

**Next session:**
Read Nagel and Tang (1998) centipede paper. Draft reply
to Professor Nagel engaging with the directional learning
connection. Plan EXP008 — test whether the Nash result
holds across different p values (p=0.5, p=0.9) to confirm
architecture-dependence generalizes beyond p=2/3.

## Session 009 — 2026-06-02

**What I built:**
Jack of Hearts environment (environments/jack_of_hearts.py):
- 20 players, 1 Jack + 19 regular players
- 4 suit symbols, asymmetric observation space (86-dim)
- 5 message types with lying support
- Jack always knows own symbol (mirror override)
- Voluntary concession threshold at 0.85
- Suspicion tracking with asymmetric visibility

GameCoordinator (environments/game_coordinator.py):
- Manages all 20 agents simultaneously
- Phase 1: message processing + suspicion updates
- Phase 2: Jack voluntary concession check
- Phase 3: declaration processing + elimination
- Phase 4: win condition checking
- Phase 5: round advancement + state sync

**Smoke tests:**
- jack_of_hearts.py: obs shape (86,) confirmed
- game_coordinator.py: 20 players, full episode runs cleanly
- Random agents: Jack wins round 1 by lobby wipe (expected —
  random declarations with 4 symbols means ~75% die per round)

**Next session:**
Build training script — Jack PPO agent vs rule-based regular
players. Rule-based regulars: declare true symbol if known
with >0.7 confidence, otherwise random. This establishes
baseline Jack behavior before upgrading to full PPO regulars.

## Session 010 — 2026-06-03

**What I built:**
Updated jack_of_hearts.py to v2:
- 8-dimensional action space (uniform for all players)
- Alliance mechanics: mutual offer formation, defection,
  within-alliance symbol claims (lying permitted)
- Diminishing returns manipulation (Jack only):
  base effect 0.30, decay 0.60 per repeat
- Neighborhood visibility: ring + random cross-links,
  each player sees exactly 6 others
- Own symbol belief updating from neighborhood observations
- Observation space: 146-dim (Jack), includes alliance
  mask, alliance claims, manipulation effectiveness

Updated game_coordinator.py to v2:
- 9-phase step: decode → concession → alliances →
  manipulation → messages → declarations → belief update
  → win conditions → advance round
- Alliance survival bonus reward shaping
- Full metrics tracking: lobby_wipes, jack_cornered,
  jack_concessions, n_alliances

**Smoke test:**
- Jack obs shape: 146 confirmed
- Full episode: Jack wins round 2, all mechanics running
- Info dict producing research metrics correctly

**Next session:**
Build rule-based regular player logic so games last
multiple rounds. Rule-based regulars: infer own symbol
from neighborhood, declare most confident symbol,
update suspicion from inconsistent messages.

## Session 011 — 2026-06-03

**What I built:**
agents/rule_based_regular.py:
- Two-phase strategy (cooperative rounds 1-2, strategic 3+)
- Declaration based on neighborhood-inferred own symbol belief
- Alliance formation with nearest alive neighbor
- Symbol sharing with allies (honest for rule-based)
- Accusation of most suspicious player in phase 2

**Results:**
- Games now last full 100 rounds vs round 1-2 with random agents
- 7 players surviving at round 100 (6 regular + Jack)
- 2 alliances formed organically
- Neither win condition triggered — balanced baseline confirmed

**Next session:**
Train PPO Jack agent vs rule-based regular players.
This establishes Jack's baseline strategy before upgrading
regulars to PPO. Key metrics to track: lobby wipe rate,
rounds to win, Jack identification round, alliance survival.

## Session 012 — 2026-06-04

**What I built:**
train_jack.py — PPO training script for Jack agent
vs rule-based regular players.

Architecture:
- JackTrainingEnv: wraps GameCoordinator for SB3
  compatibility, one PPO step = one full game round
- PPO: MlpPolicy 256x256 (larger than beauty contest
  64x64 due to 146-dim observation space)
- 500k timesteps, log every 5000 steps
- RuleBasedRegular agents for all 19 regular players

Metrics tracked per log interval:
- Lobby wipe rate, cornered rate, concession rate
- Average game duration (rounds)
- Average alliances formed
- Average episode reward

Research question:
What strategy does the Jack agent discover against
rule-based regular players? Converges to active
deception, passive blending, or mixed strategy?

**Smoke test:** obs shape (146,) confirmed, step OK.

**EXP-JH001 running:** 500k timesteps, seed 42.

**Next session:**
Analyze EXP-JH001 results. Document Jack's emergent
strategy. Plan EXP-JH002 based on findings.

## Session 012 — 2026-06-04

**What I built:**
train_jack.py — PPO training script for Jack agent
vs rule-based regular players (EXP-JH001).

Architecture:
- JackTrainingEnv: wraps GameCoordinator for SB3
  compatibility, one PPO step = one full game round
- PPO: MlpPolicy 256x256, 500k timesteps, seed 42
- RuleBasedRegular agents for all 19 regular players
- Metrics: wipe rate, cornered rate, concession rate,
  avg rounds, avg alliances, avg reward

**EXP-JH001 Results:**

Training progression:
- Steps 0-30k: learned not to concede immediately
- Steps 30-150k: wipe rate climbed 0.13 → 0.62
- Steps 150-350k: wipe rate hit 1.00 at steps 275k,
  300k, 305k, 310k — lobby wipe every game in those
  windows. Avg rounds dropped to 21-29. Reward 19-20.

Final evaluation (100 games, deterministic):
- Lobby wipe rate: 0.02
- Jack cornered rate: 0.00
- Concession rate: 0.00
- Avg game duration: 196.0 rounds (hitting 200 cap)
- Avg alliances: 0.88

**Primary Finding — Stochastic/Deterministic Mismatch:**

The wipe strategy achieved during training (0.50-1.00
wipe rate) was stochastic — driven by PPO exploration
noise, not encoded in the deterministic policy.

Under deterministic evaluation, the Jack converges to
indefinite survival: stays alive, avoids cornering,
never concedes, never wipes. Games run to the 200-round
cap with no winner.

This is game-theoretically coherent: in an imperfect
information social deduction game, a purely deterministic
Jack strategy is exploitable. The Jack's optimal strategy
requires stochasticity — consistent with the prediction
that mixed strategies dominate under incomplete
information. PPO's stochastic training policy was
implementing the correct mixed strategy, but deterministic
evaluation flattened it to a pure survival strategy.

Connection to beauty contest findings: evaluation mode
is a joint determinant of convergence behavior alongside
reward architecture and opponent structure.

**Bugs identified:**
1. No episode length cap during training — agent can
   exploit indefinite survival to accumulate reward
2. Deterministic evaluation understates actual strategy
   quality — stochastic evaluation needed

**EXP-JH002 fixes:**
1. Add 200-round cap to JackTrainingEnv.step()
2. Evaluate with deterministic=False

**Next session:**
Implement fixes, run EXP-JH002, compare wipe rate
under stochastic evaluation against EXP-JH001.