# Week 01 — 2026-05-31

## What I learned this week

The reward function design is the primary driver of agent
behavior in the p-beauty contest. Winner-take-all reward
(EXP001) doesn't produce flat training curves as predicted —
it produces Nash-seeking behavior instead. The failure mode
was more interesting than expected.

Proximity reward (EXP002) produces genuine convergence to
bounded rationality at implied level 2.726, above the Nagel
human mean of 1.8. The gap is partly population-driven
(EXP003 closes it to 2.201) and partly a training regime
effect.

## Most important failure this week

EXP005 (p=0.9) did not converge at 200k steps. Reveals that
convergence time scales with p — higher p produces a flatter
reward landscape and weaker gradient signal. This is itself
a finding about the difficulty of learning in high-p
beauty contests.

## Most important open question

Does the implied level invariance across p values (EXP002
vs EXP004 both ~2.5-2.7) hold at p=0.9 once EXP005b
converges? If yes, the agent has genuinely learned the
level-k structure. If no, there is something qualitatively
different about high-p games.