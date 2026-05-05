import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----------------- CONFIG -----------------
ENV = "single"   # "single" or "multi"

# Both summaries contain all four configs in one file
QLEARN_PATHS = {
    # "single": "/home/common/ji-bao-lin/taxi/results/q_learning/checkpoints/summary.json",
    "single": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/checkpoints/summary.json",
    "multi":  "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/checkpoints/summary.json",
}

SARSA_PATHS = {
    # "single": "/home/common/ji-bao-lin/taxi/results/sarsa/checkpoints/summary.json",
    "single": "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/checkpoints/summary.json",
    "multi":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/checkpoints/summary.json",
}

# Order: rows = decay (static/decay), columns = mask (masked/unmasked)
CONFIGS_GRID = [
    ["masked_static", "unmasked_static"],
    ["masked_decay",  "unmasked_decay"],
]

OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------


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
    header = f"{'Config':<32} {'n':>4} {'mean':>10} {'std':>8} {'median':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<32} {r['n']:>4} "
            f"{r['mean']:>10.4f} {r['std']:>8.4f} "
            f"{r['median']:>10.4f} {r['min']:>10.4f} {r['max']:>10.4f}"
        )


def stats_test(name, q_rewards, s_rewards):
    try:
        stat, p = wilcoxon(q_rewards, s_rewards)
        diff = q_rewards.mean() - s_rewards.mean()
        print(
            f"  {name:<20} Q - SARSA = {diff:+.4f}   "
            f"Wilcoxon stat={stat:.3f}, p={p:.4g}"
        )
    except ValueError as e:
        print(f"  {name}: could not compute test ({e})")


def make_2x2_plot(data_grid, env, out_path):
    """
    data_grid is a 2x2 nested list of (config_name, q_rewards, s_rewards) tuples.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    colors = ["#4C72B0", "#DD8452"]  # Q-learning, SARSA

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            config_name, q_rewards, s_rewards = data_grid[row][col]

            bp = ax.boxplot(
                [q_rewards, s_rewards],
                tick_labels=["Q-learning", "SARSA"],
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

            for i, data in enumerate([q_rewards, s_rewards], start=1):
                x_jitter = np.random.normal(i, 0.04, size=len(data))
                ax.scatter(x_jitter, data, color="black", alpha=0.6, s=20, zorder=3)

            ax.set_title(config_name)
            if col == 0:
                ax.set_ylabel("Mean episodic reward (per run)")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Q-learning vs SARSA across all four configurations — {env}-passenger taxi (30 seeds each)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    q_path = QLEARN_PATHS[ENV]
    s_path = SARSA_PATHS[ENV]

    if not Path(q_path).exists():
        sys.exit(f"Q-learning summary not found: {q_path}")
    if not Path(s_path).exists():
        sys.exit(f"SARSA summary not found: {s_path}")

    # Load all four cells
    data_grid = []
    summary_rows = []
    print(f"=== {ENV}-passenger ===\n")

    for row in CONFIGS_GRID:
        data_row = []
        for config_key in row:
            q_rewards = load_rewards(q_path, config_key)
            s_rewards = load_rewards(s_path, config_key)
            data_row.append((config_key, q_rewards, s_rewards))
            summary_rows.append(summarize(f"Q-learning  {config_key}", q_rewards))
            summary_rows.append(summarize(f"SARSA       {config_key}", s_rewards))
        data_grid.append(data_row)

    print_table(summary_rows)

    print("\n=== Q-learning vs SARSA (paired Wilcoxon, per config) ===")
    for row in data_grid:
        for config_key, q_rewards, s_rewards in row:
            stats_test(config_key, q_rewards, s_rewards)

    out_path = OUT_DIR / f"15by15_{ENV}_qlearn_vs_sarsa_2x2.png"
    make_2x2_plot(data_grid, ENV, out_path)


if __name__ == "__main__":
    main()