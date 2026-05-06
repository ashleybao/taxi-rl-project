"""
Run greedy (epsilon=0) evaluation on trained Q-tables and compare Q-learning vs SARSA. - for multi
"""

import numpy as np
import gymnasium as gym
import sys
sys.path.insert(0, "/home/common/ji-bao-lin/taxi")

# --- EDIT THESE ---------------------------------------------------------

# Path to a saved Q-table. Two formats supported:
#   - .npz with key 'q_table' shape (n_states, n_actions)  -> single best run
#   - .npy with shape (n_runs, n_states, n_actions)        -> all runs

#with no decay
# QLEARN_PATH = "/home/common/ji-bao-lin/taxi/results/q_learning/q_tables/multi_masked_qtables.npy"
# SARSA_PATH  = "/home/common/ji-bao-lin/taxi/results/sarsa/multi/multi_masked_qtables.npy"

# with decay
QLEARN_PATH = "/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/multi_masked_qtables.npy"
SARSA_PATH  = "/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/multi_masked_qtables.npy"


# Function that returns a fresh env instance — replace with your import.
# It must be the SAME env (multi-passenger, masked or unmasked) used during training.
def make_env():
    from multi_passenger_taxi import MultiPassengerTaxiEnv
    return MultiPassengerTaxiEnv()

N_EVAL_EPISODES = 500      # episodes per Q-table to evaluate
MAX_STEPS = 200            # same as your training cap
USE_ACTION_MASK = True     # True if the trained Q-table came from a masked-env run

# ------------------------------------------------------------------------


def load_q_tables(path):
    """Returns a list of Q-tables, one per run."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        return [data["q_table"]]
    elif path.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            return [arr]
        else:
            raise ValueError(f"Unexpected Q-table shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def greedy_action(q_row, action_mask=None):
    """Pick argmax action, optionally restricted to legal actions via mask."""
    if action_mask is None:
        return int(np.argmax(q_row))
    masked_q = np.where(action_mask.astype(bool), q_row, -np.inf)
    return int(np.argmax(masked_q))


def evaluate_policy(q_table, env_maker, n_episodes, max_steps, use_mask):
    """Run greedy episodes, return list of episode total rewards."""
    rewards = []
    env = env_maker()
    for ep in range(n_episodes):
        # Use episode index as eval seed so it's reproducible, NOT the training seed
        obs, info = env.reset(seed=ep)
        total = 0.0
        for _ in range(max_steps):
            mask = info.get("action_mask") if use_mask else None
            action = greedy_action(q_table[obs], mask)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return np.array(rewards)


def evaluate_all_runs(q_tables, env_maker, n_episodes, max_steps, use_mask, label):
    """Evaluate every Q-table; return per-run mean rewards."""
    per_run_means = []
    print(f"\nEvaluating {label}: {len(q_tables)} run(s) x {n_episodes} episodes")
    for i, qt in enumerate(q_tables):
        rewards = evaluate_policy(qt, env_maker, n_episodes, max_steps, use_mask)
        per_run_means.append(rewards.mean())
        print(f"  Run {i:2d}: greedy mean reward = {rewards.mean():7.3f}  "
              f"(std over episodes = {rewards.std():.3f})")
    return np.array(per_run_means)


def main():
    q_tables_qlearn = load_q_tables(QLEARN_PATH)
    q_tables_sarsa  = load_q_tables(SARSA_PATH)

    qlearn_means = evaluate_all_runs(
        q_tables_qlearn, make_env, N_EVAL_EPISODES, MAX_STEPS, USE_ACTION_MASK,
        label="Q-learning",
    )
    sarsa_means = evaluate_all_runs(
        q_tables_sarsa, make_env, N_EVAL_EPISODES, MAX_STEPS, USE_ACTION_MASK,
        label="SARSA",
    )

    print("\n=== Greedy evaluation summary ===")
    print(f"Q-learning  : mean={qlearn_means.mean():.4f}  "
          f"std={qlearn_means.std(ddof=1) if len(qlearn_means) > 1 else 0:.4f}  "
          f"n={len(qlearn_means)}")
    print(f"SARSA       : mean={sarsa_means.mean():.4f}  "
          f"std={sarsa_means.std(ddof=1) if len(sarsa_means) > 1 else 0:.4f}  "
          f"n={len(sarsa_means)}")

    if len(qlearn_means) == len(sarsa_means) and len(qlearn_means) > 1:
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(qlearn_means, sarsa_means)
            print(f"\nPaired Wilcoxon: stat={stat:.3f}, p={p:.4g}")
        except ValueError as e:
            print(f"\nCould not run Wilcoxon: {e}")


if __name__ == "__main__":
    main()