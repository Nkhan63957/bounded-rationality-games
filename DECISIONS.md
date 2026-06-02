# MAKOTO — Bounded Rationality Games
## Architectural Decisions

## Observation Space — Single-Agent vs Multi-Agent

**Decision:** Single-agent environment (beauty_contest.py) uses
a 4-feature observation [own_last_submission, last_mean,
last_target, round_progress]. Multi-agent environment
(multi_agent_beauty_contest.py) uses 3 features [last_mean,
last_target, round_progress], omitting own_last_submission.

**Rationale:** In the single-agent setting, the agent's own
previous submission carries information about its reasoning
trajectory that is not fully captured by the mean (which is
dominated by 4 fixed opponents). In the multi-agent setting,
the agent's own submission is implicitly reflected in the mean
since all 5 agents use the same policy — the mean already
encodes the agent's own behavior. The additional feature would
be redundant.

### Decision 001 — 2026-05-31

**What:** Proximity reward instead of winner-take-all as primary
training condition
**Alternatives considered:**
  - Winner-take-all (EXP001): too sparse, agent converges to
    Nash-seeking behavior rather than human-like reasoning
**Why this one:** Dense gradient signal allows agent to learn
from every round regardless of outcome. EXP001 confirmed
winner-take-all produces level 4.8 convergence (overshooting
toward Nash) rather than human-like level 1.8.
**How to reverse if wrong:** If proximity reward incentivizes
Nash play (submitting 0), switch to rank-based reward where
agent receives score proportional to placement among all players.

### Decision 002 — 2026-05-31

**What:** Mixed opponent population (40% L1, 35% L2, 15% L0,
10% L3+) as primary opponent type
**Alternatives considered:**
  - Fixed level-1 opponents (EXP003, currently running)
  - Fixed level-2 opponents
  - Pure random (level-0)
**Why this one:** Nagel (1995) human experiments used human
subjects with varying reasoning depths. Mixed population
better approximates the real experimental conditions.
**How to reverse if wrong:** If EXP003 shows significantly
closer convergence to 1.8, switch primary condition to
fixed level-1 opponents.
