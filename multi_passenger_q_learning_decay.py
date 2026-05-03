"""
Multi-Passenger Taxi — Q-Learning with Epsilon Decay
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from multi_passenger_taxi import MultiPassengerTaxiEnv

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("results/q_learning_decay/multi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BASE_RANDOM_SEED       = 58922320
N_PASSENGERS           = 2
N_RUNS                 = 12
EPISODES               = 50000
LEARNING_RATE          = 0.1
DISCOUNT_FACTOR        = 0.95
MAX_STEPS              = 200

EPSILON_START          = 1.0
EPSILON_END            = 0.01
EPSILON_DECAY_EPISODES = 40000   # decay over 80% of training


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_q_learning(
    env: MultiPassengerTaxiEnv,
    use_action_mask: bool = True,
    episodes: int = EPISODES,
    seed: int = BASE_RANDOM_SEED,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    epsilon_start: float = EPSILON_START,
    epsilon_end: float = EPSILON_END,
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES,
) -> dict:
    """Train a Q-learning agent with epsilon decay."""
    np.random.seed(seed)
    random.seed(seed)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    q_table   = np.zeros((n_states, n_actions))

    episode_rewards: list[float] = []

    for episode in range(episodes):
        # linear decay; hold at epsilon_end after decay window
        frac = min(1.0, episode / epsilon_decay_episodes)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        state, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = truncated = False

        while not (done or truncated):
            action_mask = info["action_mask"] if use_action_mask else None

            if np.random.random() < epsilon:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    action = int(np.random.choice(valid_actions))
                else:
                    action = int(np.random.randint(0, n_actions))
            else:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = int(valid_actions[np.argmax(q_table[state, valid_actions])])
                    else:
                        action = int(np.random.randint(0, n_actions))
                else:
                    action = int(np.argmax(q_table[state]))

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if not (done or truncated):
                if use_action_mask:
                    next_mask  = info["action_mask"]
                    valid_next = np.nonzero(next_mask == 1)[0]
                    next_max   = np.max(q_table[next_state, valid_next]) if len(valid_next) else 0.0
                else:
                    next_max = float(np.max(q_table[next_state]))

                q_table[state, action] += learning_rate * (
                    reward + discount_factor * next_max - q_table[state, action]
                )
            else:
                # terminal update
                q_table[state, action] += learning_rate * (
                    reward - q_table[state, action]
                )

            state = next_state

        episode_rewards.append(total_reward)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward":     float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "q_table":         q_table,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seeds = [BASE_RANDOM_SEED + i for i in range(N_RUNS)]

    masked_results_list:   list[dict] = []
    unmasked_results_list: list[dict] = []

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{N_RUNS} with seed {seed}")

        env_masked = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        masked_results = train_q_learning(
            env_masked,
            use_action_mask=True,
            episodes=EPISODES,
            seed=seed,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        )
        env_masked.close()
        masked_results_list.append(masked_results)

        env_unmasked = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        unmasked_results = train_q_learning(
            env_unmasked,
            use_action_mask=False,
            episodes=EPISODES,
            seed=seed,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        )
        env_unmasked.close()
        unmasked_results_list.append(unmasked_results)

    # Overall statistics
    masked_mean_rewards   = [r["mean_reward"] for r in masked_results_list]
    unmasked_mean_rewards = [r["mean_reward"] for r in unmasked_results_list]

    masked_overall_mean   = float(np.mean(masked_mean_rewards))
    masked_overall_std    = float(np.std(masked_mean_rewards))
    unmasked_overall_mean = float(np.mean(unmasked_mean_rewards))
    unmasked_overall_std  = float(np.std(unmasked_mean_rewards))

    # Save: summary.json
    json.dump(
        {
            "masked": [
                {"mean_reward": r["mean_reward"], "std_reward": r["std_reward"]}
                for r in masked_results_list
            ],
            "unmasked": [
                {"mean_reward": r["mean_reward"], "std_reward": r["std_reward"]}
                for r in unmasked_results_list
            ],
        },
        open(OUTPUT_DIR / "multi_summary.json", "w"),
        indent=2,
    )

    # Save: all Q-tables
    np.save(
        OUTPUT_DIR / "multi_masked_qtables.npy",
        np.array([r["q_table"] for r in masked_results_list]),
    )
    np.save(
        OUTPUT_DIR / "multi_unmasked_qtables.npy",
        np.array([r["q_table"] for r in unmasked_results_list]),
    )

    # Save: best runs
    best_idx          = max(range(N_RUNS), key=lambda i: masked_results_list[i]["mean_reward"])
    best_unmasked_idx = max(range(N_RUNS), key=lambda i: unmasked_results_list[i]["mean_reward"])

    best          = masked_results_list[best_idx]
    best_unmasked = unmasked_results_list[best_unmasked_idx]

    np.savez(
        OUTPUT_DIR / "best_multi_masked_run.npz",
        q_table         = best["q_table"],
        episode_rewards = np.array(best["episode_rewards"]),
        mean_reward     = best["mean_reward"],
        std_reward      = best["std_reward"],
        run_index       = best_idx,
        seed            = seeds[best_idx],
    )

    np.savez(
        OUTPUT_DIR / "best_multi_unmasked_run.npz",
        q_table         = best_unmasked["q_table"],
        episode_rewards = np.array(best_unmasked["episode_rewards"]),
        mean_reward     = best_unmasked["mean_reward"],
        std_reward      = best_unmasked["std_reward"],
        run_index       = best_unmasked_idx,
        seed            = seeds[best_unmasked_idx],
    )

    # Save: summary.txt
    with open(OUTPUT_DIR / "multi_summary.txt", "w") as f:
        f.write(f"N_PASSENGERS           : {N_PASSENGERS}\n")
        f.write(f"EPISODES               : {EPISODES}\n")
        f.write(f"EPSILON_START          : {EPSILON_START}\n")
        f.write(f"EPSILON_END            : {EPSILON_END}\n")
        f.write(f"EPSILON_DECAY_EPISODES : {EPSILON_DECAY_EPISODES}\n")
        f.write(f"LEARNING_RATE          : {LEARNING_RATE}\n")
        f.write(f"DISCOUNT_FACTOR        : {DISCOUNT_FACTOR}\n\n")
        f.write(f"Masked   mean: {masked_overall_mean:.4f}\n")
        f.write(f"Masked   std : {masked_overall_std:.4f}\n")
        f.write(f"Unmasked mean: {unmasked_overall_mean:.4f}\n")
        f.write(f"Unmasked std : {unmasked_overall_std:.4f}\n")

    print("\nDone. Files written to", OUTPUT_DIR)