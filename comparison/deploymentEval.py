"""
deploymentEval.py

Greedy (eps=0) evaluation of trained Q-tables across all 30 seeds.
Answers: "How well does each algorithm's policy perform when deployed?"

For each config (masked_static, unmasked_static, masked_decay, unmasked_decay)
and each environment (single-passenger, multi-passenger), evaluates all 30
Q-tables under masked greedy action selection.

Reports per-config aggregates: mean reward, success rate, episode length
across 30 seeds, plus a paired Wilcoxon test comparing Q-learning vs SARSA.
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/common/ji-bao-lin/taxi")
from multi_passenger_taxi import MultiPassengerTaxiEnv

# ----------------- CONFIG -----------------
ENV = "single"            # "single" or "multi"
N_EVAL_EPISODES = 500    # episodes per Q-table
EVAL_SEED_OFFSET = 999000
USE_MASK = True          # masked greedy argmax (correct deployment setting)

QTABLE_PATHS = {
    # ----- single-passenger -----
    ("single", "masked_static"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_masked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_masked_static_qtables.npy",
    },
    ("single", "unmasked_static"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_unmasked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_unmasked_static_qtables.npy",
    },
    ("single", "masked_decay"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_masked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_masked_decay_qtables.npy",
    },
    ("single", "unmasked_decay"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_unmasked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_unmasked_decay_qtables.npy",
    },
    # ----- 15x15 - single-passenger -----
    # ("single", "masked_static"): {
    #     "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/masked_static_qtables.npy",
    #     "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_masked_static_qtables.npy",
    # },
    # ("single", "unmasked_static"): {
    #     "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/unmasked_static_qtables.npy",
    #     "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_unmasked_static_qtables.npy",
    # },
    # ("single", "masked_decay"): {
    #     "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/masked_decay_qtables.npy",
    #     "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_masked_decay_qtables.npy",
    # },
    # ("single", "unmasked_decay"): {
    #     "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/unmasked_decay_qtables.npy",
    #     "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_unmasked_decay_qtables.npy",
    # },
    # ----- multi-passenger -----
    ("multi", "masked_static"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_masked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_masked_static_qtables.npy",
    },
    ("multi", "unmasked_static"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_unmasked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_unmasked_static_qtables.npy",
    },
    ("multi", "masked_decay"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_masked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_masked_decay_qtables.npy",
    },
    ("multi", "unmasked_decay"): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_unmasked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_unmasked_decay_qtables.npy",
    },
}

CONFIGS_TO_RUN = [
    "masked_static",
    "unmasked_static",
    "masked_decay",
    "unmasked_decay",
]

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------


def make_env():
    if ENV == "multi":
        return MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    else:
        return MultiPassengerTaxiEnv(n_passengers=1, max_steps=200)


def evaluate_q_table(q_table, n_episodes, base_seed):
    """One Q-table, N greedy episodes. Returns (mean_reward, mean_length, success_rate)."""
    env = make_env()
    rewards, lengths, successes = [], [], []
    for ep in range(n_episodes):
        state, info = env.reset(seed=base_seed + ep)
        total, steps, done, trunc = 0.0, 0, False, False
        while not (done or trunc):
            if USE_MASK:
                mask = info["action_mask"]
                valid = np.nonzero(mask == 1)[0]
                if len(valid) == 0:
                    break
                action = int(valid[np.argmax(q_table[state, valid])])
            else:
                action = int(np.argmax(q_table[state]))
            state, r, done, trunc, info = env.step(action)
            total += r
            steps += 1
        rewards.append(total)
        lengths.append(steps)
        successes.append(done and not trunc)
    env.close()
    return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(successes))


def evaluate_all_seeds(qtable_array, label):
    """Returns three arrays of length n_seeds."""
    n_seeds = qtable_array.shape[0]
    rewards, lengths, successes = [], [], []
    for i in range(n_seeds):
        r, l, s = evaluate_q_table(
            qtable_array[i],
            n_episodes=N_EVAL_EPISODES,
            base_seed=EVAL_SEED_OFFSET + i * 100000,
        )
        rewards.append(r)
        lengths.append(l)
        successes.append(s)
        print(f"  {label} seed {i:2d}: reward={r:7.2f}  length={l:5.1f}  success={s:.1%}")
    return np.array(rewards), np.array(lengths), np.array(successes)


def summarize(name, rewards, lengths, successes):
    return {
        "algorithm": name,
        "n_seeds":   len(rewards),
        "reward_mean":  float(rewards.mean()),
        "reward_std":   float(rewards.std(ddof=1)),
        "length_mean":  float(lengths.mean()),
        "length_std":   float(lengths.std(ddof=1)),
        "success_mean": float(successes.mean()),
        "success_std":  float(successes.std(ddof=1)),
    }


def run_config(config_key):
    print(f"\n{'=' * 70}")
    print(f"=== {ENV}-passenger, config: {config_key} ===")
    print(f"{'=' * 70}\n")

    paths = QTABLE_PATHS[(ENV, config_key)]
    if not Path(paths["qlearn"]).exists() or not Path(paths["sarsa"]).exists():
        print(f"  Missing Q-table file(s) for {config_key}, skipping.")
        return None

    ql_qtables = np.load(paths["qlearn"])
    sa_qtables = np.load(paths["sarsa"])

    print(f"Q-learning Q-tables: {ql_qtables.shape}")
    print(f"SARSA      Q-tables: {sa_qtables.shape}\n")

    print("Q-learning per-seed eval:")
    ql_r, ql_l, ql_s = evaluate_all_seeds(ql_qtables, "QL   ")
    print("\nSARSA per-seed eval:")
    sa_r, sa_l, sa_s = evaluate_all_seeds(sa_qtables, "SARSA")

    ql_summary = summarize("Q-learning", ql_r, ql_l, ql_s)
    sa_summary = summarize("SARSA",      sa_r, sa_l, sa_s)

    print(f"\n--- Aggregate (n={len(ql_r)}) ---")
    for s in [ql_summary, sa_summary]:
        print(f"  {s['algorithm']:<11} reward {s['reward_mean']:7.2f} ± {s['reward_std']:5.2f}   "
              f"length {s['length_mean']:6.1f} ± {s['length_std']:5.2f}   "
              f"success {s['success_mean']:.1%} ± {s['success_std']:.1%}")

    diff = ql_r - sa_r
    try:
        stat, p = wilcoxon(ql_r, sa_r)
        print(f"\n  Paired Wilcoxon (reward, Q − SARSA): "
              f"mean diff {diff.mean():+.3f}, stat={stat:.2f}, p={p:.4g}")
    except ValueError as e:
        print(f"\n  Wilcoxon failed: {e}")

    return {
        "config":     config_key,
        "env":        ENV,
        "qlearn":     ql_summary,
        "sarsa":      sa_summary,
        "ql_per_seed": {"reward": ql_r.tolist(), "length": ql_l.tolist(), "success": ql_s.tolist()},
        "sa_per_seed": {"reward": sa_r.tolist(), "length": sa_l.tolist(), "success": sa_s.tolist()},
        "wilcoxon_p": float(p) if 'p' in locals() else None,
        "mean_diff":  float(diff.mean()),
    }


def main():
    all_results = {}
    for cfg in CONFIGS_TO_RUN:
        result = run_config(cfg)
        if result is not None:
            all_results[cfg] = result

    out_path = OUT_DIR / f"deployment_eval_{ENV}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved per-seed results and aggregates to {out_path}")


if __name__ == "__main__":
    main()