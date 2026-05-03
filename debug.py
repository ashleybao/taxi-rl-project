import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# PATHS
# =========================

Q_MASKED_STATIC   = "/home/common/ji-bao-lin/taxi/masked_static_qtables.npy"
Q_UNMASKED_STATIC = "/home/common/ji-bao-lin/taxi/unmasked_static_qtables.npy"
Q_MASKED_DECAY    = "/home/common/ji-bao-lin/taxi/masked_decay_qtables.npy"

SARSA_MASKED_STATIC   = "/home/common/ji-bao-lin/taxi/results/sarsa/sarsa_masked_static_qtables.npy"
SARSA_UNMASKED_STATIC = "/home/common/ji-bao-lin/taxi/results/sarsa/sarsa_unmasked_static_qtables.npy"
SARSA_MASKED_DECAY    = "/home/common/ji-bao-lin/taxi/results/sarsa/sarsa_masked_decay_qtables.npy"

OUT_DIR = Path("results/qtable_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_LABELS = ["South", "North", "East", "West", "Pickup", "Dropoff"]

# =========================
# HELPERS
# =========================

def load(path):
    arr = np.load(path)
    print(f"Loaded {path} — shape: {arr.shape}")
    return arr


def compare(q_tables, sarsa_tables, config_name):
    print(f"\n=== {config_name} ===")

    q_mean     = q_tables.mean(axis=0)       # (n_states, n_actions)
    sarsa_mean = sarsa_tables.mean(axis=0)

    diff = np.abs(q_mean - sarsa_mean)
    print(f"  Mean abs diff : {diff.mean():.4f}")
    print(f"  Max abs diff  : {diff.max():.4f}")

    q_policy     = np.argmax(q_mean, axis=1)
    sarsa_policy = np.argmax(sarsa_mean, axis=1)
    agreement = (q_policy == sarsa_policy).mean()
    print(f"  Policy agreement: {agreement * 100:.1f}%")

    # --- heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(q_mean, aspect="auto", cmap="viridis")
    axes[0].set_title("Q-Learning (mean)")
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("State")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sarsa_mean, aspect="auto", cmap="viridis")
    axes[1].set_title("SARSA (mean)")
    axes[1].set_xlabel("Action")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, aspect="auto", cmap="hot")
    axes[2].set_title("|Q-Learning − SARSA|")
    axes[2].set_xlabel("Action")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xticks(range(len(ACTION_LABELS)))
        ax.set_xticklabels(ACTION_LABELS, rotation=45)

    fig.suptitle(f"Q-Table Comparison: Q-Learning vs SARSA — {config_name}")
    plt.tight_layout()
    out = OUT_DIR / f"qtable_diff_{config_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out}")

    # --- per-action scatter ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (ax, label) in enumerate(zip(axes.flat, ACTION_LABELS)):
        ax.scatter(q_mean[:, i], sarsa_mean[:, i], alpha=0.3, s=5)
        lo = min(q_mean[:, i].min(), sarsa_mean[:, i].min())
        hi = max(q_mean[:, i].max(), sarsa_mean[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
        ax.set_title(label)
        ax.set_xlabel("Q-Learning")
        ax.set_ylabel("SARSA")
        ax.legend()
    fig.suptitle(f"Per-Action Q-Values: Q-Learning vs SARSA — {config_name}")
    plt.tight_layout()
    out = OUT_DIR / f"qtable_scatter_{config_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out}")


# =========================
# RUN COMPARISONS
# =========================

compare(load(Q_MASKED_STATIC),   load(SARSA_MASKED_STATIC),   "Masked Static")
compare(load(Q_UNMASKED_STATIC), load(SARSA_UNMASKED_STATIC), "Unmasked Static")
compare(load(Q_MASKED_DECAY),    load(SARSA_MASKED_DECAY),    "Masked Decay")