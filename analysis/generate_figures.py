"""
Bounded Rationality Games — Figure Generation
===============================================
Produces all figures for the paper:

Beauty Contest:
  1. Experiment comparison table (all EXP001-EXP007)
  2. p-invariance plot (EXP002, EXP004, EXP005b)
  3. 2x2 architecture table visualization

Jack of Hearts:
  4. Training curves (wipe/cornered/concede over steps)
  5. EXP-JH summary metrics bar chart

Usage:
    python analysis/generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import csv

# ── Output directory ───────────────────────────────────────────

os.makedirs("docs/figures", exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
}
plt.rcParams.update(STYLE)

BLUE   = "#2E5496"
ORANGE = "#C55A11"
GREEN  = "#538135"
RED    = "#C00000"
GREY   = "#595959"


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Beauty Contest Experiment Comparison Table
# ══════════════════════════════════════════════════════════════

def fig1_experiment_table():
    experiments = [
        # (name, reward, opponents, p, implied_level, converged_at, category)
        ("EXP001", "Winner-take-all", "Mixed fixed",    "2/3", 4.806, "~50k",  "Nash"),
        ("EXP002", "Proximity",       "Mixed fixed",    "2/3", 2.726, "~150k", "Bounded"),
        ("EXP003", "Proximity",       "Level-1 fixed",  "2/3", 2.201, "~120k", "Bounded"),
        ("EXP004", "Proximity",       "Mixed fixed",    "0.5", 2.530, "~100k", "Bounded"),
        ("EXP005b","Proximity",       "Mixed fixed",    "0.9", 2.814, "~330k", "Bounded"),
        ("EXP006", "Proximity",       "Shared policy",  "2/3",10.000, "~200k", "Nash"),
        ("EXP007", "Proximity",       "Frozen sync=5",  "2/3",10.000, "~200k", "Nash"),
        ("Nagel '95","—",             "Human",          "2/3", 1.800, "one-shot","Human"),
    ]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    cols = ["Experiment", "Reward", "Opponents", "p",
            "Implied Level", "Converged At", "Outcome"]
    rows = [[e[0], e[1], e[2], e[3],
             f"{e[4]:.3f}", e[5], e[6]] for e in experiments]

    colors = []
    for e in experiments:
        if e[6] == "Nash":
            colors.append(["#FFF2CC"] * 7)
        elif e[6] == "Bounded":
            colors.append(["#E2EFDA"] * 7)
        else:
            colors.append(["#DAE8FC"] * 7)

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header styling
    for j in range(len(cols)):
        table[0, j].set_facecolor(BLUE)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Legend
    nash_patch    = mpatches.Patch(color="#FFF2CC", label="Nash-seeking")
    bounded_patch = mpatches.Patch(color="#E2EFDA", label="Bounded rationality")
    human_patch   = mpatches.Patch(color="#DAE8FC", label="Human benchmark")
    ax.legend(handles=[bounded_patch, nash_patch, human_patch],
              loc="lower right", fontsize=9, framealpha=0.8)

    ax.set_title(
        "Figure 1 — Beauty Contest Experiment Comparison\n"
        "All experiments: n=5 players, proximity reward unless noted",
        fontsize=12, fontweight="bold", pad=20, color=BLUE
    )

    plt.tight_layout()
    path = "docs/figures/fig1_experiment_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — p-Invariance Plot
# ══════════════════════════════════════════════════════════════

def fig2_p_invariance():
    # EXP004, EXP002, EXP005b
    p_values      = [0.5,   2/3,   0.9]
    implied_levels = [2.530, 2.726, 2.814]
    mean_subs     = [8.7,   16.6,  37.2]
    converged_at  = [100,   150,   330]   # thousands of steps

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A — p vs implied level
    ax = axes[0]
    ax.plot(p_values, implied_levels, "o-",
            color=BLUE, linewidth=2, markersize=9, zorder=3)
    ax.axhline(1.8, color=ORANGE, linestyle="--",
               linewidth=1.5, label="Nagel human mean (1.8)")
    ax.axhline(10.0, color=GREY, linestyle=":",
               linewidth=1.2, label="Nash equilibrium (10.0)")
    for px, ly, exp in zip(p_values, implied_levels,
                           ["EXP004\n(p=0.5)", "EXP002\n(p=2/3)",
                            "EXP005b\n(p=0.9)"]):
        ax.annotate(exp, (px, ly),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8.5, color=BLUE)
    ax.set_xlabel("p value", fontsize=11)
    ax.set_ylabel("Implied Reasoning Level", fontsize=11)
    ax.set_title("(A) Implied Level vs p\n(p-invariance)", fontsize=11)
    ax.set_ylim(0, 11)
    ax.set_xticks(p_values)
    ax.set_xticklabels(["0.5", "2/3", "0.9"])
    ax.legend(fontsize=8.5)

    # Panel B — p vs mean submission
    ax = axes[1]
    # Theoretical level-2 submissions for each p
    theoretical = [p**2 * 50 for p in p_values]
    ax.plot(p_values, mean_subs, "o-",
            color=BLUE, linewidth=2, markersize=9,
            label="PPO agent", zorder=3)
    ax.plot(p_values, theoretical, "s--",
            color=GREEN, linewidth=1.5, markersize=7,
            label="Level-2 prediction")
    ax.set_xlabel("p value", fontsize=11)
    ax.set_ylabel("Final Mean Submission", fontsize=11)
    ax.set_title("(B) Mean Submission vs p\n(scales proportionally)", fontsize=11)
    ax.set_xticks(p_values)
    ax.set_xticklabels(["0.5", "2/3", "0.9"])
    ax.legend(fontsize=8.5)

    # Panel C — p vs convergence steps
    ax = axes[2]
    ax.bar([str(p) for p in ["0.5", "2/3", "0.9"]],
           converged_at, color=[BLUE, BLUE, BLUE],
           alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("p value", fontsize=11)
    ax.set_ylabel("Steps to Convergence (thousands)", fontsize=11)
    ax.set_title("(C) Convergence Time vs p\n(scales with p)", fontsize=11)
    for i, v in enumerate(converged_at):
        ax.text(i, v + 5, f"~{v}k", ha="center",
                fontsize=9, color=BLUE, fontweight="bold")

    fig.suptitle(
        "Figure 2 — p-Invariance of Implied Reasoning Depth\n"
        "Implied level ≈ 2.5–2.8 across p=0.5, 2/3, 0.9 "
        "(consistent with level-k theory)",
        fontsize=12, fontweight="bold", color=BLUE, y=1.02
    )

    plt.tight_layout()
    path = "docs/figures/fig2_p_invariance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — 2x2 Architecture Table
# ══════════════════════════════════════════════════════════════

def fig3_architecture_table():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    data = [
        ["", "Fixed Opponents", "Same-Architecture Opponents"],
        ["Winner-take-all\nReward",
         "Nash-seeking\n(EXP001, level 4.806)",
         "[Untested]"],
        ["Proximity\nReward",
         "Bounded Rationality\n(EXP002-005, level 2.5-2.8)",
         "Nash-seeking\n(EXP006-007, level 10.0)"],
    ]

    cell_colors = [
        [BLUE,      BLUE,       BLUE],
        ["#FFF2CC", "#FFF2CC",  "#F2F2F2"],
        ["#E2EFDA", "#E2EFDA",  "#FFF2CC"],
    ]

    text_colors = [
        ["white", "white", "white"],
        ["#333", "#333",  "#888"],
        ["#333", "#333",  "#333"],
    ]

    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            rect = plt.Rectangle(
                [j/3, (2-i)/2], 1/3, 0.5,
                transform=ax.transAxes,
                facecolor=cell_colors[i][j],
                edgecolor="white", linewidth=2,
                clip_on=False
            )
            ax.add_patch(rect)
            ax.text(
                j/3 + 1/6, (2-i)/2 + 0.25,
                cell,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10,
                color=text_colors[i][j],
                fontweight="bold" if i == 0 or j == 0 else "normal",
                wrap=True
            )

    ax.set_title(
        "Figure 3 — Architecture Determinants of Convergence\n"
        "Bounded rationality requires both proximity reward "
        "AND fixed opponent distribution",
        fontsize=12, fontweight="bold", color=BLUE, pad=20
    )

    plt.tight_layout()
    path = "docs/figures/fig3_architecture_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Jack of Hearts Training Curves
# ══════════════════════════════════════════════════════════════

def fig4_joh_training_curves():
    csv_path = "experiments/working/EXP_JH001_log_20260604.csv"
    if not os.path.exists(csv_path):
        # Try JH003
        csv_path = "experiments/working/EXP_JH003_log_20260604.csv"
    if not os.path.exists(csv_path):
        print(f"  WARNING: No JH CSV found, skipping Figure 4")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A — Wipe rate over training
    ax = axes[0]
    ax.plot(df["step"]/1000, df["wipe_rate"],
            color=GREEN, linewidth=2, alpha=0.8)
    ax.fill_between(df["step"]/1000, df["wipe_rate"],
                    alpha=0.15, color=GREEN)
    ax.axhline(0.07, color=GREEN, linestyle="--",
               linewidth=1.5, label="Eval (stochastic): 0.07")
    ax.set_xlabel("Training Steps (thousands)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("(A) Lobby Wipe Rate\n(Jack wins)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)

    # Panel B — Cornered rate over training
    ax = axes[1]
    ax.plot(df["step"]/1000, df["corner_rate"],
            color=RED, linewidth=2, alpha=0.8)
    ax.fill_between(df["step"]/1000, df["corner_rate"],
                    alpha=0.15, color=RED)
    ax.axhline(0.00, color=RED, linestyle="--",
               linewidth=1.5, label="Eval (stochastic): 0.00")
    ax.set_xlabel("Training Steps (thousands)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("(B) Jack Cornered Rate\n(Regular players win)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)

    # Panel C — Concession rate over training
    ax = axes[2]
    ax.plot(df["step"]/1000, df["concede_rate"],
            color=ORANGE, linewidth=2, alpha=0.8)
    ax.fill_between(df["step"]/1000, df["concede_rate"],
                    alpha=0.15, color=ORANGE)
    ax.axhline(0.05, color=ORANGE, linestyle="--",
               linewidth=1.5, label="Eval (stochastic): 0.05")
    ax.set_xlabel("Training Steps (thousands)", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("(C) Voluntary Concession Rate\n(Jack strategic concession)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)

    fig.suptitle(
        "Figure 4 — Jack of Hearts Training Curves (EXP-JH001)\n"
        "Jack PPO agent vs rule-based regular players, 500k timesteps",
        fontsize=12, fontweight="bold", color=BLUE, y=1.02
    )

    plt.tight_layout()
    path = "docs/figures/fig4_joh_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 5 — Jack of Hearts Summary Metrics
# ══════════════════════════════════════════════════════════════

def fig5_joh_summary():
    experiments = ["EXP-JH001\n(no cap,\ndeterministic)",
                   "EXP-JH002\n(cap,\ndeterministic)",
                   "EXP-JH003\n(cap,\nstochastic)"]
    wipe_rates   = [0.02, 0.07, 0.07]
    corner_rates = [0.00, 0.00, 0.00]
    avg_durations = [196.0, 185.5, 185.5]
    avg_alliances = [0.88,  1.29,  1.29]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A — Wipe and corner rates
    ax = axes[0]
    x = np.arange(len(experiments))
    w = 0.35
    bars1 = ax.bar(x - w/2, wipe_rates, w,
                   label="Wipe rate", color=GREEN, alpha=0.85)
    bars2 = ax.bar(x + w/2, corner_rates, w,
                   label="Cornered rate", color=RED, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(0, 0.20)
    ax.set_title("(A) Win Condition Rates\nAcross Experiment Variants",
                 fontsize=11)
    ax.legend(fontsize=9)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.2f}", ha="center", fontsize=8.5,
                    color=GREEN, fontweight="bold")

    # Panel B — Avg duration and alliances
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(x - w/2, avg_durations, w,
           label="Avg duration (rounds)", color=BLUE, alpha=0.8)
    ax2.bar(x + w/2, avg_alliances, w,
            label="Avg alliances", color=ORANGE, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.set_ylabel("Avg Game Duration (rounds)", fontsize=11, color=BLUE)
    ax2.set_ylabel("Avg Alliances Formed", fontsize=11, color=ORANGE)
    ax.set_title("(B) Game Duration and Alliance Formation\n"
                 "Across Experiment Variants", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.suptitle(
        "Figure 5 — Jack of Hearts Evaluation Summary\n"
        "Stochastic evaluation reveals mixed strategy "
        "equilibrium (EXP-JH003)",
        fontsize=12, fontweight="bold", color=BLUE, y=1.02
    )

    plt.tight_layout()
    path = "docs/figures/fig5_joh_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...\n")
    fig1_experiment_table()
    fig2_p_invariance()
    fig3_architecture_table()
    fig4_joh_training_curves()
    fig5_joh_summary()
    print("\nAll figures saved to docs/figures/")