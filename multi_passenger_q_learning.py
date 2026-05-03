"""
Multi-Passenger Taxi — Q-Learning Training Script
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from multi_passenger_taxi import MultiPassengerTaxiEnv

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BASE_RANDOM_SEED = 58922320
N_PASSENGERS     = 2
N_RUNS           = 30          # 30 runs recommended for statistical significance tests
EPISODES         = 20000
LEARNING_RATE    = 0.1
DISCOUNT_FACTOR  = 0.95
EPSILON          = 0.1
MAX_STEPS        = 200

output_dir = Path("results/q_learning/multi_passenger")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoints_dir = Path("results/q_learning/multi_passenger/checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

qtables_dir = Path("results/q_learning/multi_passenger/q_tables")
qtables_dir.mkdir(parents=True, exist_ok=True)

rewards_dir = Path("results/q_learning/multi_passenger/rewards")
rewards_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# epsilon SCHEDULES
# ---------------------------------------------------------------------------

def linear_decay_schedule_factory(episodes, eps_start=1.0, eps_end=0.05, decay_frac=0.8):
    decay_episodes = int(episodes * decay_frac)

    def schedule(episode):
        if episode < decay_episodes:
            return eps_start - (eps_start - eps_end) * (episode / decay_episodes)
        return eps_end

    return schedule


def exponential_decay_schedule_factory(eps_start=1.0, eps_end=0.05, decay_rate=0.999):
    def schedule(episode):
        return max(eps_end, eps_start * (decay_rate ** episode))
    return schedule


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
    epsilon: float = EPSILON,
    epsilon_schedule=None,
    track_epsilon: bool = False,
) -> dict:
    """Train a Q-learning agent with or without action masking.

    Returns
    -------
    dict with keys:
        episode_rewards : list[float]
        mean_reward     : float
        std_reward      : float
        q_table         : np.ndarray  shape (n_states, n_actions)
        epsilons        : list[float] | None
    """
    np.random.seed(seed)
    random.seed(seed)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    q_table   = np.zeros((n_states, n_actions))

    episode_rewards: list[float] = []
    epsilon_trace = [] if track_epsilon else None

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = truncated = False

        current_epsilon = epsilon_schedule(episode) if epsilon_schedule else epsilon

        if track_epsilon:
            epsilon_trace.append(current_epsilon)

        while not (done or truncated):
            action_mask = info["action_mask"] if use_action_mask else None

            # ---- Epsilon-greedy action selection ----
            if np.random.random() < current_epsilon:
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

            # ---- Q-learning update ----
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
                # terminal update — bootstrap with 0
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
        "epsilons":        epsilon_trace,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seeds = [BASE_RANDOM_SEED + i for i in range(N_RUNS)]

    linear_schedule = linear_decay_schedule_factory(EPISODES)

    masked_static:   list[dict] = []
    unmasked_static: list[dict] = []
    masked_decay:    list[dict] = []
    unmasked_decay:  list[dict] = []

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{N_RUNS} with seed {seed}")

        # ---- Masked static ----
        env = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        masked_static.append(train_q_learning(
            env, use_action_mask=True, episodes=EPISODES, seed=seed,
            learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON,
        ))
        env.close()

        # ---- Unmasked static ----
        env = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        unmasked_static.append(train_q_learning(
            env, use_action_mask=False, episodes=EPISODES, seed=seed,
            learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON,
        ))
        env.close()

        # ---- Masked decay ----
        env = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        masked_decay.append(train_q_learning(
            env, use_action_mask=True, episodes=EPISODES, seed=seed,
            learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
            epsilon_schedule=linear_schedule, track_epsilon=True,
        ))
        env.close()

        # ---- Unmasked decay ----
        env = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=MAX_STEPS)
        unmasked_decay.append(train_q_learning(
            env, use_action_mask=False, episodes=EPISODES, seed=seed,
            learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
            epsilon_schedule=linear_schedule, track_epsilon=True,
        ))
        env.close()

    # -----------------------------------------------------------------------
    # Overall statistics
    # -----------------------------------------------------------------------
    def overall_stats(results):
        means = [r["mean_reward"] for r in results]
        return float(np.mean(means)), float(np.std(means))

    masked_static_mean,   masked_static_std   = overall_stats(masked_static)
    unmasked_static_mean, unmasked_static_std = overall_stats(unmasked_static)
    masked_decay_mean,    masked_decay_std    = overall_stats(masked_decay)
    unmasked_decay_mean,  unmasked_decay_std  = overall_stats(unmasked_decay)

    # -----------------------------------------------------------------------
    # Analysis helpers
    # -----------------------------------------------------------------------

    def mean_curve(results):
        return np.mean([r["episode_rewards"] for r in results], axis=0)

    def smooth(x, w=200):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    masked_static_curve   = mean_curve(masked_static)
    unmasked_static_curve = mean_curve(unmasked_static)
    masked_decay_curve    = mean_curve(masked_decay)
    unmasked_decay_curve  = mean_curve(unmasked_decay)

    # -----------------------------------------------------------------------
    # Plot 1 — all four curves
    # -----------------------------------------------------------------------

    plt.figure(figsize=(12, 7))
    plt.plot(smooth(masked_static_curve),   label="Masked (Static eps)",     alpha=0.9)
    plt.plot(smooth(unmasked_static_curve), label="Unmasked (Static eps)",   alpha=0.9)
    plt.plot(smooth(masked_decay_curve),    label="Masked (Decaying eps)",   alpha=0.9)
    plt.plot(smooth(unmasked_decay_curve),  label="Unmasked (Decaying eps)", alpha=0.9)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title(f"Q-Learning: Exploration Strategy & Masking - Multi-Passenger Taxi (n={N_PASSENGERS})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "multi_passenger_learning_curve.png", dpi=150, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Plot 2 — epsilon schedule effect
    # -----------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(smooth(masked_decay_curve),   label="Masked (Decaying eps)",   alpha=0.9)
    ax1.plot(smooth(unmasked_decay_curve), label="Unmasked (Decaying eps)", alpha=0.9)
    ax1.set_ylabel("Smoothed Reward")
    ax1.set_title(f"Effect of Decaying eps Schedule - Q-Learning Multi-Passenger (n={N_PASSENGERS})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    mean_epsilon_trace = np.mean(
        [r["epsilons"] for r in masked_decay if r["epsilons"] is not None], axis=0
    )
    ax2.plot(mean_epsilon_trace, color="darkorange", label="eps value")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon (eps)")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multi_passenger_epsilon_schedule_effect.png", dpi=150, bbox_inches="tight")

    # -----------------------------------------------------------------------
    # Save: summary.json
    # -----------------------------------------------------------------------
    json.dump(
        {
            "masked_static": [
                {"mean_reward": r["mean_reward"], "std": r["std_reward"]}
                for r in masked_static
            ],
            "unmasked_static": [
                {"mean_reward": r["mean_reward"], "std": r["std_reward"]}
                for r in unmasked_static
            ],
            "masked_decay": [
                {"mean_reward": r["mean_reward"], "std": r["std_reward"]}
                for r in masked_decay
            ],
            "unmasked_decay": [
                {"mean_reward": r["mean_reward"], "std": r["std_reward"]}
                for r in unmasked_decay
            ],
        },
        open(checkpoints_dir / "summary.json", "w"),
        indent=2,
    )

    # -----------------------------------------------------------------------
    # Save: per-seed reward arrays (shape: N_RUNS x EPISODES)
    # for statistical significance testing
    # -----------------------------------------------------------------------
    np.save(
        rewards_dir / "masked_static_rewards.npy",
        np.array([r["episode_rewards"] for r in masked_static]),
    )
    np.save(
        rewards_dir / "unmasked_static_rewards.npy",
        np.array([r["episode_rewards"] for r in unmasked_static]),
    )
    np.save(
        rewards_dir / "masked_decay_rewards.npy",
        np.array([r["episode_rewards"] for r in masked_decay]),
    )
    np.save(
        rewards_dir / "unmasked_decay_rewards.npy",
        np.array([r["episode_rewards"] for r in unmasked_decay]),
    )

    # -----------------------------------------------------------------------
    # Save: all Q-tables  (shape: N_RUNS x n_states x n_actions)
    # -----------------------------------------------------------------------
    np.save(qtables_dir / "multi_masked_static_qtables.npy",   np.array([r["q_table"] for r in masked_static]))
    np.save(qtables_dir / "multi_unmasked_static_qtables.npy", np.array([r["q_table"] for r in unmasked_static]))
    np.save(qtables_dir / "multi_masked_decay_qtables.npy",    np.array([r["q_table"] for r in masked_decay]))
    np.save(qtables_dir / "multi_unmasked_decay_qtables.npy",  np.array([r["q_table"] for r in unmasked_decay]))

    # -----------------------------------------------------------------------
    # Save: best runs (include episode_rewards for significance testing)
    # -----------------------------------------------------------------------
    def save_best(results, seeds, path):
        idx  = max(range(N_RUNS), key=lambda i: results[i]["mean_reward"])
        best = results[idx]
        np.savez(
            path,
            q_table         = best["q_table"],
            episode_rewards = np.array(best["episode_rewards"]),
            mean_reward     = best["mean_reward"],
            std_reward      = best["std_reward"],
            run_index       = idx,
            seed            = seeds[idx],
        )

    save_best(masked_static,   seeds, checkpoints_dir / "best_multi_masked_static.npz")
    save_best(unmasked_static, seeds, checkpoints_dir / "best_multi_unmasked_static.npz")
    save_best(masked_decay,    seeds, checkpoints_dir / "best_multi_masked_decay.npz")
    save_best(unmasked_decay,  seeds, checkpoints_dir / "best_multi_unmasked_decay.npz")

    # -----------------------------------------------------------------------
    # Save: summary.txt
    # -----------------------------------------------------------------------
    with open(checkpoints_dir / "summary.txt", "w") as f:
        f.write(f"N_PASSENGERS    : {N_PASSENGERS}\n")
        f.write(f"N_RUNS          : {N_RUNS}\n")
        f.write(f"EPISODES        : {EPISODES}\n")
        f.write(f"EPSILON         : {EPSILON} (static) / linear decay (decay runs)\n")
        f.write(f"LEARNING_RATE   : {LEARNING_RATE}\n")
        f.write(f"DISCOUNT_FACTOR : {DISCOUNT_FACTOR}\n\n")
        f.write(f"Masked   static mean: {masked_static_mean:.4f}  std: {masked_static_std:.4f}\n")
        f.write(f"Unmasked static mean: {unmasked_static_mean:.4f}  std: {unmasked_static_std:.4f}\n")
        f.write(f"Masked   decay  mean: {masked_decay_mean:.4f}  std: {masked_decay_std:.4f}\n")
        f.write(f"Unmasked decay  mean: {unmasked_decay_mean:.4f}  std: {unmasked_decay_std:.4f}\n")

    print("\nDone. Files written:")
    print(f"  {output_dir}/multi_passenger_learning_curve.png")
    print(f"  {output_dir}/multi_passenger_epsilon_schedule_effect.png")
    print(f"  {checkpoints_dir}/summary.json")
    print(f"  {checkpoints_dir}/summary.txt")
    print(f"  {checkpoints_dir}/best_multi_masked_static.npz")
    print(f"  {checkpoints_dir}/best_multi_unmasked_static.npz")
    print(f"  {checkpoints_dir}/best_multi_masked_decay.npz")
    print(f"  {checkpoints_dir}/best_multi_unmasked_decay.npz")
    print(f"  {qtables_dir}/multi_masked_static_qtables.npy")
    print(f"  {qtables_dir}/multi_unmasked_static_qtables.npy")
    print(f"  {qtables_dir}/multi_masked_decay_qtables.npy")
    print(f"  {qtables_dir}/multi_unmasked_decay_qtables.npy")
    print(f"  {rewards_dir}/masked_static_rewards.npy       # shape: ({N_RUNS}, {EPISODES})")
    print(f"  {rewards_dir}/unmasked_static_rewards.npy     # shape: ({N_RUNS}, {EPISODES})")
    print(f"  {rewards_dir}/masked_decay_rewards.npy        # shape: ({N_RUNS}, {EPISODES})")
    print(f"  {rewards_dir}/unmasked_decay_rewards.npy      # shape: ({N_RUNS}, {EPISODES})")