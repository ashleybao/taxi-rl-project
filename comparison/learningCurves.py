import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
ENV = "single"  # "single" or "multi"

SIZE = "15by15"  # "5by5" or "15by15"

REWARD_DIRS = {
    "single": {
        # "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/rewards", 
        # "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/rewards",
        "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/rewards",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/rewards",
    },
    "multi": {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/rewards",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/rewards",
    },
}

CONFIGS = ["masked_static", "unmasked_static", "masked_decay", "unmasked_decay"]
SMOOTH_WINDOW = 500

OUT_DIR = Path("plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------


def smooth_rows(x, w=SMOOTH_WINDOW):
    """Rolling-mean smoothing applied to each row."""
    kernel = np.ones(w) / w
    return np.array([np.convolve(row, kernel, mode="valid") for row in x])


def plot_one_config(ax, ql, sarsa, title):
    """Plot Q-learning vs SARSA with shaded ± std band across seeds."""
    ql_s    = smooth_rows(ql)
    sarsa_s = smooth_rows(sarsa)

    x = np.arange(ql_s.shape[1])

    ql_mean    = ql_s.mean(axis=0)
    ql_std     = ql_s.std(axis=0)
    sarsa_mean = sarsa_s.mean(axis=0)
    sarsa_std  = sarsa_s.std(axis=0)

    ax.plot(x, ql_mean, label="Q-learning", color="#4C72B0")
    ax.fill_between(x, ql_mean - ql_std, ql_mean + ql_std, color="#4C72B0", alpha=0.2)

    ax.plot(x, sarsa_mean, label="SARSA", color="#DD8452")
    ax.fill_between(x, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, color="#DD8452", alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (rolling mean, 500-ep)")
    ax.grid(alpha=0.3)
    ax.legend()


def main():
    ql_dir    = Path(REWARD_DIRS[ENV]["qlearn"])
    sarsa_dir = Path(REWARD_DIRS[ENV]["sarsa"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    config_grid = [
        ["masked_static",   "unmasked_static"],
        ["masked_decay",    "unmasked_decay"],
    ]

    for row in range(2):
        for col in range(2):
            cfg = config_grid[row][col]
            ql    = np.load(ql_dir    / f"{cfg}_rewards.npy")
            sarsa = np.load(sarsa_dir / f"{cfg}_rewards.npy")
            plot_one_config(axes[row, col], ql, sarsa, cfg)

    fig.suptitle(
        f"{SIZE}_Learning curves — {ENV}-passenger taxi (mean ± std across 30 seeds)",
        fontsize=14,
    )
    plt.tight_layout()
    out_path = OUT_DIR / f"{SIZE}_{ENV}_learning_curves_2x2.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()