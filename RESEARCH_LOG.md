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
