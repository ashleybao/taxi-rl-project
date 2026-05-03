import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import json
from bigger_taxi_env import BiggerTaxiEnv

# Base random seed for reproducibility
BASE_RANDOM_SEED = 58922320

output_dir = Path("results/15by15map/sarsa")
output_dir.mkdir(parents=True, exist_ok=True)
# =========================
# ε SCHEDULES
# =========================

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


# =========================
# SARSA
# =========================
def train_sarsa(
    env,
    use_action_mask=True,
    episodes=5000,
    seed=BASE_RANDOM_SEED,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1,
    epsilon_schedule=None,
    track_epsilon=False,
):
    np.random.seed(seed)
    random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    episode_rewards = []
    epsilon_trace = [] if track_epsilon else None

    def select_action(state, info, epsilon):
        action_mask = info["action_mask"] if use_action_mask else None

        if np.random.random() < epsilon:
            if use_action_mask:
                valid_actions = np.nonzero(action_mask == 1)[0]
                return np.random.choice(valid_actions)
            return np.random.randint(0, n_actions)
        else:
            if use_action_mask:
                valid_actions = np.nonzero(action_mask == 1)[0]
                return valid_actions[np.argmax(q_table[state, valid_actions])]
            return np.argmax(q_table[state])

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        current_epsilon = epsilon_schedule(episode) if epsilon_schedule else epsilon

        if track_epsilon:
            epsilon_trace.append(current_epsilon)

        action = select_action(state, info, current_epsilon)

        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if not (done or truncated):
                next_action = select_action(next_state, info, current_epsilon)

                q_table[state, action] += learning_rate * (
                    reward
                    + discount_factor * q_table[next_state, next_action]
                    - q_table[state, action]
                )

                state = next_state
                action = next_action
            else:
                # terminal update (optional but clean)
                q_table[state, action] += learning_rate * (
                    reward - q_table[state, action]
                )

        episode_rewards.append(total_reward)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "q_table": q_table,
        "epsilons": epsilon_trace,
    }

# =========================
# EXPERIMENT SETUP
# =========================

n_runs = 12
episodes = 5000
learning_rate = 0.1
discount_factor = 0.95

seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]

masked_static = []
unmasked_static = []
masked_decay = []
unmasked_decay = []

linear_schedule = linear_decay_schedule_factory(episodes)

# =========================
# RUN EXPERIMENTS
# =========================

for i, seed in enumerate(seeds):
    print(f"Run {i + 1}/{n_runs} seed {seed}")

    # -------- static masking --------
    env = BiggerTaxiEnv(size=15)
    masked_static.append(
        train_sarsa(
            env,
            use_action_mask=True,
            seed=seed,
            episodes=episodes,
            epsilon=0.1,
        )
    )
    env.close()

    # -------- static unmasked --------
    env = BiggerTaxiEnv(size=15)
    unmasked_static.append(
        train_sarsa(
            env,
            use_action_mask=False,
            seed=seed,
            episodes=episodes,
            epsilon=0.1,
        )
    )
    env.close()

    # -------- decaying masking --------
    env = BiggerTaxiEnv(size=15)
    masked_decay.append(
        train_sarsa(
            env,
            use_action_mask=True,
            seed=seed,
            episodes=episodes,
            epsilon_schedule=linear_schedule,
            track_epsilon=True,
        )
    )
    env.close()

    # -------- decaying unmasked --------
    env = BiggerTaxiEnv(size=15)
    unmasked_decay.append(
        train_sarsa(
            env,
            use_action_mask=False,
            seed=seed,
            episodes=episodes,
            epsilon_schedule=linear_schedule,
            track_epsilon=True,
        )
    )
    env.close()


# =========================
# ANALYSIS HELPERS
# =========================

def mean_curve(results):
    return np.mean([r["episode_rewards"] for r in results], axis=0)


def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


masked_static_curve = mean_curve(masked_static)
unmasked_static_curve = mean_curve(unmasked_static)
masked_decay_curve = mean_curve(masked_decay)
unmasked_decay_curve = mean_curve(unmasked_decay)


# =========================
# PLOT
# =========================

plt.figure(figsize=(12, 7))

plt.plot(smooth(masked_static_curve), label="Masked (Static ε)", alpha=0.9)
plt.plot(smooth(unmasked_static_curve), label="Unmasked (Static ε)", alpha=0.9)
plt.plot(smooth(masked_decay_curve), label="Masked (Decaying ε)", alpha=0.9)
plt.plot(smooth(unmasked_decay_curve), label="Unmasked (Decaying ε)", alpha=0.9)

plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Exploration Strategy and Action Masking Effects (Taxi-v3)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(output_dir / "sarsa_epsilon_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================
# SAVE RESULTS
# =========================

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
    },
    open(output_dir / "summary.json", "w"),
)


np.save(output_dir / "sarsa_masked_static_qtables.npy", [r["q_table"] for r in masked_static])
np.save(output_dir / "sarsa_unmasked_static_qtables.npy", [r["q_table"] for r in unmasked_static])
np.save(output_dir / "sarsa_masked_decay_qtables.npy", [r["q_table"] for r in masked_decay])
np.save(output_dir / "sarsa_unmasked_decay_qtables.npy", [r["q_table"] for r in unmasked_decay])

# =========================
# BEST RUNS
# =========================

best_masked = max(masked_static, key=lambda r: r["mean_reward"])
best_unmasked = max(unmasked_static, key=lambda r: r["mean_reward"])
best_decay = max(masked_decay, key=lambda r: r["mean_reward"])

np.savez(
    output_dir / "sarsa_best_masked_static.npz",
    q_table=best_masked["q_table"],
    mean_reward=best_masked["mean_reward"],
)

np.savez(
    output_dir / "sarsa_best_unmasked_static.npz",
    q_table=best_unmasked["q_table"],
    mean_reward=best_unmasked["mean_reward"],
)

np.savez(
    output_dir / "sarsa_best_masked_decay.npz",
    q_table=best_decay["q_table"],
    mean_reward=best_decay["mean_reward"],
)