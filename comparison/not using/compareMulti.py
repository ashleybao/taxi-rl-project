import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

CONFIG_KEY = "unmasked"

# # with decay 
# QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/multi_summary.json"
# SARSA = "/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/multi_summary.json"
# OUT_PLOT = f"multi_qlearn_vs_sarsa_boxplot_decay_{CONFIG_KEY}.png"

# no decay 
QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning/summary/multi_summary.json"
SARSA = "/home/common/ji-bao-lin/taxi/results/sarsa/multi/multi_summary.json"
OUT_PLOT = f"multi_qlearn_vs_sarsa_boxplot_{CONFIG_KEY}.png"

def load_rewards(path, config_key=CONFIG_KEY):
    with open(path) as f:
        data = json.load(f)
    if config_key not in data:
        raise KeyError(
            f"'{config_key}' not found in {path}. Available keys: {list(data.keys())}"
        )
    return np.array([run["mean_reward"] for run in data[config_key]])


def summarize(name, rewards):
    print(f"--- {name} ---")
    print(f"  n runs : {len(rewards)}")
    print(f"  mean   : {rewards.mean():.4f}")
    print(f"  std    : {rewards.std(ddof=1):.4f}")
    print(f"  median : {np.median(rewards):.4f}")
    print(f"  min    : {rewards.min():.4f}")
    print(f"  max    : {rewards.max():.4f}")
    print(f"  range  : {rewards.max() - rewards.min():.4f}")


def stats_test(q_rewards, s_rewards):
    """Run paired test - used the same base random seed"""
    print("\n--- Statistical tests ---")

    if len(q_rewards) == len(s_rewards):
        try:
            stat, p = wilcoxon(q_rewards, s_rewards)
            print(f"  Wilcoxon signed-rank (paired): stat={stat:.3f}, p={p:.4g}")
        except ValueError as e:
            # wilcoxon errors when all differences are zero
            print(f"  Wilcoxon signed-rank: could not compute ({e})")

    diff = q_rewards.mean() - s_rewards.mean()
    print(f"  Mean difference (Q - SARSA)   : {diff:+.4f}")


def make_boxplot(q_rewards, s_rewards, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(
        [q_rewards, s_rewards],
        labels=["Q-learning", "SARSA"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = ["#4C72B0", "#DD8452"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)


    for i, data in enumerate([q_rewards, s_rewards], start=1):
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, color="black", alpha=0.6, s=20, zorder=3)

    ax.set_ylabel("Mean episodic reward (per run)")
    ax.set_title(f"[With decay epsilon] Q-learning vs SARSA — {CONFIG_KEY}, multi-passenger taxi\n(12 seeds each)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved box plot to: {out_path}")


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
            f"Pass paths as args: python compare_qlearn_sarsa.py qlearn.json sarsa.json"
        )

    config = CONFIG_KEY
    q_rewards = load_rewards(q_path, config)
    s_rewards = load_rewards(s_path, config)

    print(f"=== Comparing config: '{config}' ===\n")
    summarize("Q-learning", q_rewards)
    print()
    summarize("SARSA", s_rewards)

    stats_test(q_rewards, s_rewards)
    make_boxplot(q_rewards, s_rewards, OUT_PLOT)


if __name__ == "__main__":
    main()