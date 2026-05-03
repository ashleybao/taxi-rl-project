import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

CONFIG_KEY = "multi"  # or "single"
MASKED_KEY = "masked_static" # "masked_static" for single, "masked" for multi

# for multi - no decay
# QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning/summary/multi_summary.json"
# SARSA = "/home/common/ji-bao-lin/taxi/results/sarsa/multi/multi_summary.json"
# OUT_PLOT = "multi_masked_vs_unmasked_boxplots.png"

# with decay 
# QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/multi_summary.json"
# SARSA = "/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/multi_summary.json"
# OUT_PLOT = "multi_masked_vs_unmasked_boxplots_decay.png"


# # for single
QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning/summary/summary.json"
SARSA = "/home/common/ji-bao-lin/taxi/results/sarsa/summary.json"
OUT_PLOT = "single_masked_vs_unmasked_boxplots.png"


def load_rewards(path, config_key):
    with open(path) as f:
        data = json.load(f)
    if config_key not in data:
        raise KeyError(
            f"'{config_key}' not found in {path}. Available keys: {list(data.keys())}"
        )
    return np.array([run["mean_reward"] for run in data[config_key]])


def summarize(name, rewards):
    return {
        "name": name,
        "n": len(rewards),
        "mean": rewards.mean(),
        "std": rewards.std(ddof=1),
        "median": float(np.median(rewards)),
        "min": rewards.min(),
        "max": rewards.max(),
    }


def print_table(rows):
    header = f"{'Config':<25} {'n':>4} {'mean':>10} {'std':>8} {'median':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<25} {r['n']:>4} "
            f"{r['mean']:>10.4f} {r['std']:>8.4f} "
            f"{r['median']:>10.4f} {r['min']:>10.4f} {r['max']:>10.4f}"
        )


def masking_effect_test(name, masked, unmasked):
    try:
        stat, p = wilcoxon(masked, unmasked)
        diff = masked.mean() - unmasked.mean()
        print(
            f"  {name:<12} masked - unmasked = {diff:+.4f}   "
            f"Wilcoxon stat={stat:.3f}, p={p:.4g}"
        )
    except ValueError as e:
        print(f"  {name}: could not compute test ({e})")


def make_side_by_side_plot(q_m, q_u, s_m, s_u, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    panels = [
        ("Q-learning", q_m, q_u, axes[0]),
        ("SARSA", s_m, s_u, axes[1]),
    ]
    colors = ["#4C72B0", "#DD8452"]  # masked, unmasked

    for title, masked, unmasked, ax in panels:
        bp = ax.boxplot(
            [masked, unmasked],
            tick_labels=["Masked", "Unmasked"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        for i, data in enumerate([masked, unmasked], start=1):
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, color="black", alpha=0.6, s=20, zorder=3)

        ax.set_title(title)
        ax.set_ylabel("Mean episodic reward (per run)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-300, -30)

    fig.suptitle(
        f"[With decay epsilon] Effect of action masking — {CONFIG_KEY}-passenger taxi (12 seeds each)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")

def main():
    if len(sys.argv) >= 3:
        q_path, s_path = sys.argv[1], sys.argv[2]
    else:
        q_path, s_path = QLEARN, SARSA
        print(f"(No args given — using defaults: {q_path}, {s_path})")

    if not Path(q_path).exists() or not Path(s_path).exists():
        sys.exit(
            f"Could not find input files. Looked for:\n"
            f"  {q_path}\n  {s_path}\n"
            f"Pass paths as args: python compareEnv.py qlearn.json sarsa.json"
        )

    # Load all four cells of the 2x2
    q_masked = load_rewards(q_path, f"{MASKED_KEY}")
    q_unmasked = load_rewards(q_path, f"un{MASKED_KEY}")
    s_masked = load_rewards(s_path, f"{MASKED_KEY}")
    s_unmasked = load_rewards(s_path, f"un{MASKED_KEY}")

    # Summary table
    rows = [
        summarize("Q-learning  masked", q_masked),
        summarize("Q-learning  unmasked", q_unmasked),
        summarize("SARSA       masked", s_masked),
        summarize("SARSA       unmasked", s_unmasked),
    ]
    print("=== 2x2 summary ===")
    print_table(rows)

    # Masking effect within each algorithm
    print("\n=== Masking effect (paired Wilcoxon, masked vs unmasked) ===")
    masking_effect_test("Q-learning", q_masked, q_unmasked)
    masking_effect_test("SARSA", s_masked, s_unmasked)

    # Plot
    make_side_by_side_plot(q_masked, q_unmasked, s_masked, s_unmasked, OUT_PLOT)


if __name__ == "__main__":
    main()