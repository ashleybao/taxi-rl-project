"""
Compute masked-argmax policy agreement and Q-value spread between Q-learning and SARSA, across all 30 seeds.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, "/home/common/ji-bao-lin/taxi")
from multi_passenger_taxi import MultiPassengerTaxiEnv

# ----------------- CONFIG -----------------
ENV = "single"           # "single" or "multi"
DECAY = True            
N_ROLLOUT_EPISODES = 200 
EVAL_SEED_OFFSET = 999000 


QTABLE_PATHS = {
    ("single", False): {
        # "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_masked_static_qtables.npy",
        # "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_masked_static_qtables.npy",
        "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/masked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_masked_static_qtables.npy",
    },
    ("single", True): {
        # "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/q_learning_masked_decay_qtables.npy",
        # "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/q_tables/sarsa_masked_decay_qtables.npy",
        "qlearn": "/home/common/ji-bao-lin/taxi/results/15by15map/q_learning/q_tables/masked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/15by15map/sarsa/q_tables/sarsa_masked_decay_qtables.npy",
    },
    ("multi", False): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_masked_static_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_masked_static_qtables.npy",
    },
    ("multi", True): {
        "qlearn": "/home/common/ji-bao-lin/taxi/results/q_learning/multi_passenger/q_tables/multi_masked_decay_qtables.npy",
        "sarsa":  "/home/common/ji-bao-lin/taxi/results/sarsa/multi_passenger/q_tables/multi_masked_decay_qtables.npy",
    },
}
# ------------------------------------------


def make_env():
    if ENV == "multi":
        return MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    else:
        return MultiPassengerTaxiEnv(n_passengers=1, max_steps=200)


def collect_states_and_compare(q_ql, q_sarsa, n_episodes, base_seed):
    """For one (QL, SARSA) Q-table pair, roll out QL greedy, compare argmax at each visited state."""
    env = make_env()
    state_to_mask = {}

    for ep in range(n_episodes):
        state, info = env.reset(seed=base_seed + ep)
        state_to_mask[state] = info["action_mask"]
        done = trunc = False
        while not (done or trunc):
            valid = np.nonzero(info["action_mask"] == 1)[0]
            if len(valid) == 0:
                break
            action = int(valid[np.argmax(q_ql[state, valid])])
            state, _, done, trunc, info = env.step(action)
            if not (done or trunc):
                state_to_mask[state] = info["action_mask"]
    env.close()

    agree = 0
    total = 0
    qval_diffs = []
    for s, mask in state_to_mask.items():
        valid = np.nonzero(mask == 1)[0]
        if len(valid) == 0:
            continue
        a_ql    = int(valid[np.argmax(q_ql[s, valid])])
        a_sarsa = int(valid[np.argmax(q_sarsa[s, valid])])
        total += 1
        if a_ql == a_sarsa:
            agree += 1
        qval_diffs.append(abs(q_ql[s, a_ql] - q_sarsa[s, a_sarsa]))

    return total, agree, np.array(qval_diffs)


def main():
    paths = QTABLE_PATHS[(ENV, DECAY)]
    ql_path = Path(paths["qlearn"])
    sa_path = Path(paths["sarsa"])

    if not ql_path.exists():
        sys.exit(f"Q-learning Q-tables not found: {ql_path}")
    if not sa_path.exists():
        sys.exit(f"SARSA Q-tables not found: {sa_path}")

    ql_qtables = np.load(ql_path)     # (n_seeds, n_states, n_actions)
    sa_qtables = np.load(sa_path)

    print(f"Q-learning Q-tables shape: {ql_qtables.shape}")
    print(f"SARSA      Q-tables shape: {sa_qtables.shape}")
    assert ql_qtables.shape == sa_qtables.shape, "Q-table shapes don't match!"

    n_seeds = ql_qtables.shape[0]
    print(f"\nComparing {n_seeds} paired seeds, {N_ROLLOUT_EPISODES} rollout episodes each")
    print(f"Config: ENV={ENV}, DECAY={DECAY}\n")

    per_seed_agreement = []
    per_seed_qval_diff = []
    per_seed_total = []

    for i in range(n_seeds):
        total, agree, qdiffs = collect_states_and_compare(
            ql_qtables[i], sa_qtables[i],
            n_episodes=N_ROLLOUT_EPISODES,
            base_seed=EVAL_SEED_OFFSET + i * 100000,
        )
        agreement = agree / total if total else float("nan")
        mean_qdiff = qdiffs.mean() if len(qdiffs) else float("nan")
        per_seed_agreement.append(agreement)
        per_seed_qval_diff.append(mean_qdiff)
        per_seed_total.append(total)
        print(f"  Seed {i:2d}: states={total:5d}  agreement={agreement:.1%}  "
              f"mean |Q_QL-Q_SARSA|={mean_qdiff:.3f}")

    per_seed_agreement = np.array(per_seed_agreement)
    per_seed_qval_diff = np.array(per_seed_qval_diff)

    print(f"\n=== Aggregate across {n_seeds} seeds ===")
    print(f"Policy agreement : mean = {per_seed_agreement.mean():.1%}  "
          f"std = {per_seed_agreement.std(ddof=1):.1%}  "
          f"min = {per_seed_agreement.min():.1%}  max = {per_seed_agreement.max():.1%}")
    print(f"Q-value spread   : mean = {per_seed_qval_diff.mean():.3f}  "
          f"std = {per_seed_qval_diff.std(ddof=1):.3f}  "
          f"min = {per_seed_qval_diff.min():.3f}  max = {per_seed_qval_diff.max():.3f}")
    print(f"States visited   : mean = {np.mean(per_seed_total):.0f}  "
          f"min = {min(per_seed_total)}  max = {max(per_seed_total)}")


if __name__ == "__main__":
    main()