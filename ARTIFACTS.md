# MAKOTO — Artifact Manifest & Next Steps

This maps everything produced in this session to its role, whether to keep or
retire it, and where it lands in the planned 3-file architecture. The through
-line: **every reported number lives on a neutral metric (real welfare), is
benchmarked against an analytic game-theoretic solution, and is written to a
JSON by one writer.** Nothing is reported in a shaped/amplified reward unit.

---

## 1. KEEP — code (becomes the core of `pipeline.py`)

| File | What it is | Role in final architecture |
|---|---|---|
| `trade_game.py` | Validated two-country tariff game (optimal-tariff theory). Analytic Nash + cooperative benchmarks; numerically proven prisoner's dilemma (reaction slope −0.30, Nash Pareto-dominated). | `pipeline.py` → **GAME** section. The analytic Nash/cooperative are the benchmarks every learner is scored against. |
| `repeated_game.py` | Repeated simultaneous-move tariff env (`RepeatedTariffEnv`) + `CooperativePlannerEnv`. Reward = own welfare in neutral units. | `pipeline.py` → **ENVIRONMENT** section. Drop-in target for PPO in Colab. |

## 2. KEEP — results & figures (paper exhibits + dashboard inputs)

| File | Finding it carries | Where it goes |
|---|---|---|
| `qre_results.json` | The bounded-rationality sweep data. | **Single-writer JSON**: pipeline writes it, dashboard reads it. The de-hardcoding rule starts here. |
| `qre_experiment.png` | Core positive result: rational learners → Nash; boundedly-rational learners play a QRE that trade-wars *harder* than Nash and never reaches cooperation. | Paper Figure (Results). |
| `trade_game_validation.png` | The game is a genuine PD (reaction curves, welfare surface, coordination gap 0.237). | Paper Figure (Methods/Model). |
| `decomposition.png` | Why the OLD headline was an artifact: +2.941 = −0.139 real policy + 3.080 reward-scale. | Paper Figure (Methods justification — *this is why we evaluate on neutral welfare*). |
| `coordination_gap.png` | Why the OLD multi-agent game was inert: cooperation gap ≈ 0 at spillover 0.03; the additive structure had no strategic teeth. | Paper Figure (motivates switching to the tariff game). |

## 3. RETIRE — claims that don't survive scrutiny

These come from the existing repo and the v8 paper. Do not carry them forward.

- **README "+12.185 ± 1.390" multi-agent advantage** — reward-scale artifact, not strategic structure.
- **"Structurally irreversible advantage" / "smoking gun" ablation framing** (pipeline docstrings, v8 paper) — the ablation shows the opposite: MAKOTO's policy is −0.139 *worse* on a neutral scoreboard.
- **`mechanism_confirmed` ρ=0.829, p=0.042 docstring** — the code reproduces ρ=−0.034, p=0.879 ("Mechanism confirmed: NO"). Remove the false number.
- **Hardcoded dashboard values** ($18.8B, $94.0B, CV 0.1363, recall tables, the seed-42 90.618/79.537 row that the live run contradicts).

## 4. KEEP — reusable infrastructure (already in repo, no claim attached)

Detector (IsolationForest), LSTM ensemble + walk-forward CV, data-fetch layer,
multi-seed harness, bootstrap CI machinery. These are sound and reused; they
just feed a different (neutral, benchmarked) evaluation now.

---

## Immediate next steps

1. **You own the calibration.** Source US/China bilateral trade elasticities
   and pin `beta, kappa, delta` (and make them asymmetric — US vs China differ
   in market size / terms-of-trade power). Optimal-tariff theory: `beta/kappa`
   ≈ inverse foreign export-supply elasticity. This is the fix for your GRC
   "researcher-set multipliers" point — replace defaults with sourced numbers.
2. **PPO confirmation (Colab/GPU).** Run two independent PPO agents through
   `RepeatedTariffEnv` and confirm they land near the QRE we characterized
   analytically. The harness is ready; only the learner swaps in.
3. **Repeated-game cooperation experiment.** Let partners condition on history
   (reciprocity). Folk-theorem question: can a repeated structure sustain
   cooperation below Nash? This is the natural sequel to "independent learning
   never reaches cooperation."
4. **Embed in the real panel** (2000–2023) for out-of-sample, rolling-origin
   evaluation across regimes — the honest OOS layer.

## What to be able to explain in an interview (own these)

- Why the tariff game is a real prisoner's dilemma and the old spillover was not.
- Why QRE (not Nash) is the right prediction for learning agents, and what λ means.
- Why bounded rationality *widens* the coordination gap here.
- Why every metric is neutral welfare, not shaped reward — and the decomposition that forced that choice.
