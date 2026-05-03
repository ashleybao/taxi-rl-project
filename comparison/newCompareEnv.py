import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----------------- CONFIG -----------------
ENV = "single"   # "single" or "multi"
DECAY = False    # False -> compare static (no-decay) cells; True -> compare decay cells

# Q-learning paths (one file per environment, contains all four configs)
QLEARN_PATHS = {
    "single": "/home/common/ji-bao-lin/taxi/results/q_learning/checkpoints/summary.json",
    "multi":  "/home/common/ji-bao-lin/taxi/results/q_learning/summary/multi_summary.json",  # update if path is different
}

# SARSA paths: two files per environment, masked/unmasked keys inside
SARSA_PATHS = {
    ("single", False): "/home/common/ji-bao-lin/taxi/results/sarsa/summary.json",
    ("single", True):  "/home/common/ji-bao-lin/taxi/results/sarsa_decay/summary.json",
    ("multi",  False): "/home/common/ji-bao-lin/taxi/results/sarsa/multi/multi_summary.json",
    ("multi",  True):  "/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/multi_summary.json",
}

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


def make_side_by_side_plot(q_m, q_u, s_m, s_u, env, decay, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    panels = [
        ("Q-learning", q_m, q_u, axes[0]),
        ("SARSA", s_m, s_u, axes[1]),
    ]
    colors = ["#4C72B0", "#DD8452"]

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

    decay_label = "with decay epsilon" if decay else "no decay epsilon"
    fig.suptitle(
        f"[{decay_label}] Effect of action masking — {env}-passenger taxi (12 seeds each)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


def main():
    q_path = QLEARN_PATHS[ENV]
    s_path = SARSA_PATHS[(ENV, DECAY)]

    if not Path(q_path).exists():
        sys.exit(f"Q-learning summary not found: {q_path}")
    if not Path(s_path).exists():
        sys.exit(f"SARSA summary not found: {s_path}")

    # Q-learning has all four configs in one file. Pick the masked/unmasked pair for the chosen DECAY setting.
    q_mask_key   = "masked_decay"   if DECAY else "masked_static"
    q_unmask_key = "unmasked_decay" if DECAY else "unmasked_static"

    # SARSA file already split by decay/non-decay; keys inside are just "masked"/"unmasked"
    s_mask_key   = "masked"
    s_unmask_key = "unmasked"

    q_masked   = load_rewards(q_path, q_mask_key)
    q_unmasked = load_rewards(q_path, q_unmask_key)
    s_masked   = load_rewards(s_path, s_mask_key)
    s_unmasked = load_rewards(s_path, s_unmask_key)

    print(f"=== {ENV}-passenger, decay={DECAY} ===\n")
    rows = [
        summarize("Q-learning  masked",   q_masked),
        summarize("Q-learning  unmasked", q_unmasked),
        summarize("SARSA       masked",   s_masked),
        summarize("SARSA       unmasked", s_unmasked),
    ]
    print_table(rows)

    print("\n=== Masking effect (paired Wilcoxon, masked vs unmasked) ===")
    masking_effect_test("Q-learning", q_masked, q_unmasked)
    masking_effect_test("SARSA",      s_masked, s_unmasked)

    decay_tag = "decay" if DECAY else "static"
    out_path = OUT_DIR / f"{ENV}_masked_vs_unmasked_boxplots_{decay_tag}.png"
    make_side_by_side_plot(q_masked, q_unmasked, s_masked, s_unmasked, ENV, DECAY, out_path)


if __name__ == "__main__":
    main()