"""
For each state s, the policy picks argmax_a Q(s,a). This script asks:
  - In how many states do the two algorithms pick the same action?
  - Where do they disagree, and what kind of disagreements are they?
  - Is the agreement consistent across the 30 seeds?

Inputs: two .npy files of shape (n_runs, n_states, n_actions) — typically
        multi_masked_qtables.npy for Q-learning and SARSA.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


QLEARN_PATH = "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/multi_masked_qtables.npy"
SARSA_PATH  = "/home/common/ji-bao-lin/taxi/results/sarsa/multi/multi_masked_qtables.npy"

OUT_PLOT = "policy_agreement.png"

# Action indices for the multi-passenger taxi env. Adjust if your env uses
# a different mapping. Standard Taxi-v3 mapping:
#   0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff
ACTION_NAMES = ["South", "North", "East", "West", "Pickup", "Dropoff"]
MOVEMENT_ACTIONS = {0, 1, 2, 3}
INTERACTION_ACTIONS = {4, 5}
# ------------------------------------------------------------------------


def load_qtables(path):
    arr = np.load(path)
    if arr.ndim != 3:
        sys.exit(f"Expected (n_runs, n_states, n_actions); got shape {arr.shape}")
    return arr  # (n_runs, n_states, n_actions)


def per_pair_agreement(q_tables_a, q_tables_b):
    """For each (run_a_i, run_b_i) pair, % of states where argmax matches."""
    n_runs = q_tables_a.shape[0]
    assert q_tables_b.shape[0] == n_runs, "Different number of runs in the two files"
    rates = []
    for i in range(n_runs):
        pol_a = q_tables_a[i].argmax(axis=1)
        pol_b = q_tables_b[i].argmax(axis=1)
        rate = (pol_a == pol_b).mean()
        rates.append(rate)
    return np.array(rates)


def aggregate_disagreement_matrix(q_tables_a, q_tables_b):
    """6x6 matrix counting disagreements: M[i,j] = # of (state, run) pairs where
       algo A picked i and algo B picked j. Diagonal = agreements."""
    n_actions = q_tables_a.shape[2]
    M = np.zeros((n_actions, n_actions), dtype=np.int64)
    for run in range(q_tables_a.shape[0]):
        pol_a = q_tables_a[run].argmax(axis=1)
        pol_b = q_tables_b[run].argmax(axis=1)
        # Vectorised count
        for a, b in zip(pol_a, pol_b):
            M[a, b] += 1
    return M


def category_breakdown(M):
    """Group disagreements by movement vs pickup/dropoff."""
    n = M.shape[0]
    total = M.sum()
    diag = np.trace(M)
    move_idx = list(MOVEMENT_ACTIONS)
    inter_idx = list(INTERACTION_ACTIONS)

    # Both picked movement (possibly different directions)
    both_move = M[np.ix_(move_idx, move_idx)].sum()
    move_agree = sum(M[i, i] for i in move_idx)
    move_disagree = both_move - move_agree

    # Both picked pickup/dropoff
    both_inter = M[np.ix_(inter_idx, inter_idx)].sum()
    inter_agree = sum(M[i, i] for i in inter_idx)
    inter_disagree = both_inter - inter_agree

    # One movement, one pickup/dropoff (the most "consequential" disagreement)
    cross = total - both_move - both_inter

    print("\n=== Disagreement breakdown ===")
    print(f"  Total state-run cells     : {total}")
    print(f"  Agreements                : {diag} ({diag/total:.1%})")
    print(f"  Disagreements             : {total-diag} ({(total-diag)/total:.1%})")
    print(f"    - both picked move, but different direction : {move_disagree}")
    print(f"    - both picked pickup/dropoff but different  : {inter_disagree}")
    print(f"    - one movement, one pickup/dropoff          : {cross}  <-- biggest behavioral gap")


def plot_results(per_run_rates, M, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: per-run agreement bar chart
    ax = axes[0]
    runs = np.arange(len(per_run_rates))
    ax.bar(runs, per_run_rates * 100, color="#4C72B0", alpha=0.8)
    ax.axhline(per_run_rates.mean() * 100, color="black", linestyle="--",
               label=f"mean = {per_run_rates.mean()*100:.1f}%")
    ax.set_xlabel("Seed (paired Q-learning vs SARSA run)")
    ax.set_ylabel("% of states with same argmax action")
    ax.set_title("Policy agreement per seed")
    ax.set_ylim(0, 100)
    ax.set_xticks(runs)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: action confusion matrix
    ax = axes[1]
    M_norm = M / M.sum() * 100  # percent of total
    im = ax.imshow(M_norm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_yticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("SARSA action")
    ax.set_ylabel("Q-learning action")
    ax.set_title("Action confusion matrix (% of all state-run pairs)")
    for i in range(len(ACTION_NAMES)):
        for j in range(len(ACTION_NAMES)):
            val = M_norm[i, j]
            color = "white" if val > M_norm.max() / 2 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")


def main():
    if not Path(QLEARN_PATH).exists() or not Path(SARSA_PATH).exists():
        sys.exit("Could not find one of the .npy files. Edit the paths at the top.")

    q_tables_qlearn = load_qtables(QLEARN_PATH)
    q_tables_sarsa  = load_qtables(SARSA_PATH)

    print(f"Q-learning Q-tables: shape {q_tables_qlearn.shape}")
    print(f"SARSA      Q-tables: shape {q_tables_sarsa.shape}")

    # 1) Agreement rate per paired run (seed i vs seed i)
    rates = per_pair_agreement(q_tables_qlearn, q_tables_sarsa)
    print("\n=== Per-seed agreement (paired) ===")
    for i, r in enumerate(rates):
        print(f"  Seed {i:2d}: {r*100:6.2f}% of states have same argmax action")
    print(f"\n  Across-seed mean agreement: {rates.mean()*100:.2f}%")
    print(f"  Across-seed std           : {rates.std(ddof=1)*100:.2f}%")
    print(f"  Min / Max                 : {rates.min()*100:.2f}% / {rates.max()*100:.2f}%")

    # 2) Aggregate confusion matrix across all runs and states
    M = aggregate_disagreement_matrix(q_tables_qlearn, q_tables_sarsa)

    # 3) Category breakdown
    category_breakdown(M)

    # 4) Plots
    plot_results(rates, M, OUT_PLOT)


if __name__ == "__main__":
    main()