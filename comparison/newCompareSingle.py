import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----------------- CONFIG -----------------
# Both summaries contain all four configs in one file:
#   masked_static, unmasked_static, masked_decay, unmasked_decay
# QLEARN = "/home/common/ji-bao-lin/taxi/results/q_learning/checkpoints/summary.json"
# SARSA  = "/home/common/ji-bao-lin/taxi/results/sarsa/checkpoints/summary.json"  # <-- update if path differs

# 15x15
QLEARN = "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/checkpoints/summary.json"
SARSA  = "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/checkpoints/summary.json"  

CONFIGS_TO_RUN = [
    "masked_static",
    "unmasked_static",
    "masked_decay",
    "unmasked_decay",
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
    print(f"--- {name} ---")
    print(f"  n runs : {len(rewards)}")
    print(f"  mean   : {rewards.mean():.4f}")
    print(f"  std    : {rewards.std(ddof=1):.4f}")
    print(f"  median : {np.median(rewards):.4f}")
    print(f"  min    : {rewards.min():.4f}")
    print(f"  max    : {rewards.max():.4f}")
    print(f"  range  : {rewards.max() - rewards.min():.4f}")


def stats_test(q_rewards, s_rewards):
    print("\n--- Statistical tests ---")
    if len(q_rewards) == len(s_rewards):
        try:
            stat, p = wilcoxon(q_rewards, s_rewards)
            print(f"  Wilcoxon signed-rank (paired): stat={stat:.3f}, p={p:.4g}")
        except ValueError as e:
            print(f"  Wilcoxon signed-rank: could not compute ({e})")
    diff = q_rewards.mean() - s_rewards.mean()
    print(f"  Mean difference (Q - SARSA)   : {diff:+.4f}")


def make_boxplot(q_rewards, s_rewards, config_key, out_path):
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
    ax.set_title(
        f"Q-learning vs SARSA — {config_key}\n"
        f"single-passenger taxi (30 seeds each)"
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved box plot to: {out_path}")


def run_one(config_key):
    print(f"\n{'=' * 60}")
    print(f"=== Comparing config: '{config_key}' ===")
    print(f"{'=' * 60}\n")

    q_rewards = load_rewards(QLEARN, config_key)

    if not Path(SARSA).exists():
        print(f"(SARSA source missing, plotting Q-learning only)")
        summarize("Q-learning", q_rewards)
        return
    s_rewards = load_rewards(SARSA, config_key)

    summarize("Q-learning", q_rewards)
    print()
    summarize("SARSA", s_rewards)

    stats_test(q_rewards, s_rewards)

    out_path = OUT_DIR / f"15by15_single_qlearn_vs_sarsa_boxplot_{config_key}.png"
    make_boxplot(q_rewards, s_rewards, config_key, out_path)


def main():
    if not Path(QLEARN).exists():
        sys.exit(f"Q-learning summary not found at: {QLEARN}")

    for key in CONFIGS_TO_RUN:
        run_one(key)


if __name__ == "__main__":
    main()