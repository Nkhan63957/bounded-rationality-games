# Bounded Rationality Games

Computational study of strategic reasoning under uncertainty 
in two game-theoretic environments:

**Case 1 — p-Beauty Contest (Keynesian Beauty Contest)**  
Players submit a number in [0, 100]. Winner is closest to 
p × mean(all submissions). Tests whether PPO agents exhibit 
bounded rationality consistent with human subjects (Nagel 1995).

**Case 2 — Adversarial Information Game (Jack of Hearts)**  
N players must identify their own type from others' signals,
with one adversarial agent actively providing false information.
Tests Byzantine fault tolerance under identity-unknown adversary.

## Key Results (Beauty Contest)

| Experiment | p | Opponents | Implied Level |
|---|---|---|---|
| EXP002 | 2/3 | Mixed | 2.726 |
| EXP003 | 2/3 | Level-1 | 2.201 |
| EXP004 | 0.5 | Mixed | 2.530 |
| EXP005b | 0.9 | Mixed | pending |
| **Nagel (1995) humans** | **2/3** | **Human** | **~1.8** |

## Structure
environments/   — Game environments
agents/         — PPO and level-k agents
analysis/       — Results analysis
experiments/    — Saved models (working/failed)
docs/           — Screenshots and weekly summaries

## Research Log
See RESEARCH_LOG.md for full session-by-session documentation
including failure modes and architectural decisions.