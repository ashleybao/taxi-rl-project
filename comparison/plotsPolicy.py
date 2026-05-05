import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paste your per-seed data manually as lists, OR re-run greedyCheck saving them to JSON
# For cleaner code, modify greedyCheck.py to save these lists to JSON

DATA = {
    "single_static_5x5": {
        "agreement": [88.6, 89.2, 91.0, 88.3, 90.2, 88.6, 88.8, 89.1, 86.7, 91.8,
                      86.7, 86.1, 84.4, 87.9, 81.8, 88.5, 85.3, 86.2, 84.1, 88.8,
                      87.3, 90.7, 87.8, 88.7, 82.3, 90.2, 88.7, 81.4, 86.2, 85.5],
        "qval":      [2.609, 2.632, 2.604, 2.662, 2.656, 2.645, 2.578, 2.694, 2.590, 2.575,
                      2.669, 2.534, 2.649, 2.630, 2.670, 2.724, 2.548, 2.625, 2.642, 2.503,
                      2.654, 2.675, 2.667, 2.674, 2.713, 2.608, 2.636, 2.602, 2.621, 2.595],
    },
    "single_decay_5x5": {
        "agreement": [80.0, 83.4, 86.1, 84.4, 85.7, 81.8, 81.9, 85.9, 84.2, 84.0,
                      84.4, 84.6, 84.7, 85.5, 83.2, 86.9, 83.7, 85.7, 86.3, 83.4,
                      84.0, 85.4, 83.5, 83.1, 83.5, 81.4, 87.3, 85.1, 85.4, 77.8],
        "qval":      [6.473, 6.465, 6.094, 6.354, 6.404, 6.393, 6.327, 6.170, 5.975, 6.199,
                      6.304, 6.257, 6.474, 6.309, 6.396, 6.049, 6.323, 6.302, 6.241, 6.395,
                      6.557, 6.262, 6.428, 6.110, 6.394, 6.520, 6.472, 6.357, 6.309, 6.513],
    },
    "multi_static_5x5": {
        "agreement": [89.7, 89.4, 89.5, 88.4, 87.8, 89.1, 89.0, 88.5, 88.7, 89.7,
                      88.7, 88.7, 88.6, 89.5, 88.1, 88.6, 88.7, 88.5, 87.8, 89.5,
                      88.2, 88.0, 88.1, 88.7, 89.1, 89.5, 89.2, 88.0, 88.8, 89.5],
        "qval":      [3.221, 3.145, 3.164, 3.144, 3.093, 3.146, 3.139, 3.091, 3.186, 3.196,
                      3.233, 3.070, 3.221, 3.079, 3.110, 3.192, 3.148, 3.074, 3.134, 3.168,
                      3.078, 3.223, 3.085, 3.197, 3.149, 3.097, 3.079, 3.202, 3.109, 3.174],
    },
    "multi_decay_5x5": {
        "agreement": [84.3, 83.2, 83.9, 84.2, 83.9, 85.3, 84.9, 83.7, 83.9, 82.9,
                      85.1, 84.3, 84.8, 84.6, 84.2, 85.4, 84.4, 84.4, 84.5, 84.5,
                      84.2, 83.5, 82.8, 85.0, 82.5, 83.5, 84.5, 83.9, 84.2, 83.8],
        "qval":      [5.690, 5.720, 5.822, 5.707, 5.654, 5.670, 5.637, 5.569, 5.730, 5.820,
                      5.722, 5.613, 5.610, 5.746, 5.775, 5.758, 5.625, 5.601, 5.797, 5.704,
                      5.634, 5.843, 5.601, 5.873, 5.819, 5.669, 5.568, 5.660, 5.547, 5.514],
    },
    "single_static_15x15": {
        "agreement": [92.6, 89.5, 93.4, 91.9, 92.5, 92.9, 93.2, 93.5, 91.6, 94.2,
                      91.8, 93.8, 89.6, 92.6, 92.4, 92.9, 91.9, 93.8, 90.6, 92.6,
                      92.2, 90.9, 91.6, 94.7, 94.3, 94.2, 92.8, 92.2, 94.2, 93.3],
        "qval":      [0.121, 0.097, 0.093, 0.082, 0.146, 0.092, 0.114, 0.082, 0.092, 0.112,
                      0.103, 0.136, 0.084, 0.079, 0.071, 0.088, 0.124, 0.090, 0.094, 0.110,
                      0.077, 0.099, 0.079, 0.105, 0.088, 0.085, 0.079, 0.074, 0.079, 0.103],
    },
    "single_decay_15x15": {
        "agreement": [92.5, 90.9, 93.5, 92.6, 94.5, 90.8, 90.9, 93.0, 92.2, 92.9,
                      92.7, 94.2, 91.6, 93.0, 92.1, 93.2, 91.9, 92.5, 93.6, 93.1,
                      91.1, 92.7, 92.5, 93.0, 90.9, 92.1, 92.1, 92.1, 92.7, 92.0],
        "qval":      [0.236, 0.242, 0.220, 0.249, 0.300, 0.273, 0.282, 0.227, 0.208, 0.286,
                      0.253, 0.267, 0.266, 0.310, 0.280, 0.240, 0.300, 0.255, 0.282, 0.253,
                      0.261, 0.252, 0.308, 0.271, 0.294, 0.286, 0.254, 0.277, 0.255, 0.284],
    },
}

ORDER = ["single_static_5x5", "single_decay_5x5",
         "multi_static_5x5", "multi_decay_5x5",
         "single_static_15x15", "single_decay_15x15"]
LABELS = ["Single 5×5\nStatic ε", "Single 5×5\nDecay ε",
          "Multi 5×5\nStatic ε", "Multi 5×5\nDecay ε",
          "Single 15×15\nStatic ε", "Single 15×15\nDecay ε"]
COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#CCB974"]

Path("plots").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Policy agreement
agreement_data = [DATA[k]["agreement"] for k in ORDER]
bp1 = axes[0].boxplot(agreement_data, tick_labels=LABELS, patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=1.5))
for patch, c in zip(bp1["boxes"], COLORS):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for i, data in enumerate(agreement_data, start=1):
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    axes[0].scatter(x_jitter, data, color="black", alpha=0.6, s=15, zorder=3)
axes[0].set_ylabel("Policy agreement (%)")
axes[0].set_title("Masked-greedy policy agreement (Q-learning vs SARSA)")
axes[0].tick_params(axis="x", labelsize=8)
axes[0].grid(axis="y", alpha=0.3)

# Panel 2: Q-value spread
qval_data = [DATA[k]["qval"] for k in ORDER]
bp2 = axes[1].boxplot(qval_data, tick_labels=LABELS, patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=1.5))
for patch, c in zip(bp2["boxes"], COLORS):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for i, data in enumerate(qval_data, start=1):
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    axes[1].scatter(x_jitter, data, color="black", alpha=0.6, s=15, zorder=3)
axes[1].set_ylabel("Mean |Q_QL - Q_SARSA| at chosen action")
axes[1].set_title("Q-value spread")
axes[1].tick_params(axis="x", labelsize=8)
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Policy and value comparison across configurations (30 seeds each)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/policy_qval_comparison.png", dpi=150)
print("saved plots/policy_qval_comparison.png")

fig, ax = plt.subplots(figsize=(9, 6))
for k, label, c in zip(ORDER, LABELS, COLORS):
    ax.scatter(DATA[k]["qval"], DATA[k]["agreement"], label=label.replace("\n", " "),
               color=c, alpha=0.7, s=50)
ax.set_xlabel("Q-value spread")
ax.set_ylabel("Policy agreement (%)")
ax.set_title("Per-seed (n=30) policy agreement vs Q-value spread")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/agreement_vs_qspread_scatter.png", dpi=150)
print("saved plots/agreement_vs_qspread_scatter.png")