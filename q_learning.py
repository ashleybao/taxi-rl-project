import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import json

# Base random seed for reproducibility
BASE_RANDOM_SEED = 58922320

output_dir = Path("results/q_learning")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoints_dir = Path("results/q_learning/checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

qtables_dir = Path("results/q_learning/q_tables")
qtables_dir.mkdir(parents=True, exist_ok=True)

rewards_dir = Path("results/q_learning/rewards")
rewards_dir.mkdir(parents=True, exist_ok=True)


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
# Q-LEARNING
# =========================

def train_q_learning(
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

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        total_reward = 0
        done = False
        truncated = False

        current_epsilon = epsilon_schedule(episode) if epsilon_schedule else epsilon

        if track_epsilon:
            epsilon_trace.append(current_epsilon)

        while not (done or truncated):

            action_mask = info["action_mask"] if use_action_mask else None

            # epsilon-greedy
            if np.random.random() < current_epsilon:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, n_actions)
            else:
                if use_action_mask:
                    valid_actions = np.nonzero(action_mask == 1)[0]
                    action = valid_actions[np.argmax(q_table[state, valid_actions])]
                else:
                    action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if not (done or truncated):
                if use_action_mask:
                    next_mask = info["action_mask"]
                    valid_next_actions = np.nonzero(next_mask == 1)[0]
                    next_max = (
                        np.max(q_table[next_state, valid_next_actions])
                        if len(valid_next_actions) > 0
                        else 0
                    )
                else:
                    next_max = np.max(q_table[next_state])

                q_table[state, action] += learning_rate * (
                    reward + discount_factor * next_max - q_table[state, action]
                )

            else:
                q_table[state, action] += learning_rate * (reward - q_table[state, action])

            state = next_state

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

n_runs = 30          # 30 runs recommended for statistical significance tests
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
    env = gym.make("Taxi-v3")
    masked_static.append(
        train_q_learning(
            env,
            use_action_mask=True,
            seed=seed,
            episodes=episodes,
            epsilon=0.1,
        )
    )
    env.close()

    # -------- static unmasked --------
    env = gym.make("Taxi-v3")
    unmasked_static.append(
        train_q_learning(
            env,
            use_action_mask=False,
            seed=seed,
            episodes=episodes,
            epsilon=0.1,
        )
    )
    env.close()

    # -------- decaying masking --------
    env = gym.make("Taxi-v3")
    masked_decay.append(
        train_q_learning(
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
    env = gym.make("Taxi-v3")
    unmasked_decay.append(
        train_q_learning(
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
# PLOT 1 — all four curves
# =========================

plt.figure(figsize=(12, 7))

plt.plot(smooth(masked_static_curve), label="Masked (Static ε)", alpha=0.9)
plt.plot(smooth(unmasked_static_curve), label="Unmasked (Static ε)", alpha=0.9)
plt.plot(smooth(masked_decay_curve), label="Masked (Decaying ε)", alpha=0.9)
plt.plot(smooth(unmasked_decay_curve), label="Unmasked (Decaying ε)", alpha=0.9)

plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Effect of ε Decay vs Static Exploration (Taxi-v3)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(output_dir / "epsilon_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# =========================
# PLOT 2 — epsilon schedule effect
# =========================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(smooth(masked_decay_curve), label="Masked (Decaying ε)", alpha=0.9)
ax1.plot(smooth(unmasked_decay_curve), label="Unmasked (Decaying ε)", alpha=0.9)
ax1.set_ylabel("Smoothed Reward")
ax1.set_title("Effect of Decaying ε Schedule - Q-Learning")
ax1.legend()
ax1.grid(True, alpha=0.3)

mean_epsilon_trace = np.mean(
    [r["epsilons"] for r in masked_decay if r["epsilons"] is not None], axis=0
)
ax2.plot(mean_epsilon_trace, color="darkorange", label="ε value")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon (ε)")
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "q_learning_epsilon_schedule_effect.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================
# SAVE RESULTS — summary.json
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
        "unmasked_decay": [
            {"mean_reward": r["mean_reward"], "std": r["std_reward"]}
            for r in unmasked_decay
        ],
    },
    open(checkpoints_dir / "summary.json", "w"),
)


# =========================
# SAVE — per-seed reward arrays (shape: n_runs × episodes)
# for statistical significance testing
# =========================

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


# =========================
# SAVE — all Q-tables (shape: n_runs × n_states × n_actions)
# =========================

np.save(qtables_dir / "q_learning_masked_static_qtables.npy",   [r["q_table"] for r in masked_static])
np.save(qtables_dir / "q_learning_unmasked_static_qtables.npy", [r["q_table"] for r in unmasked_static])
np.save(qtables_dir / "q_learning_masked_decay_qtables.npy",    [r["q_table"] for r in masked_decay])
np.save(qtables_dir / "q_learning_unmasked_decay_qtables.npy",  [r["q_table"] for r in unmasked_decay])


# =========================
# SAVE — best runs (include episode_rewards for significance testing)
# =========================

best_masked        = max(masked_static,   key=lambda r: r["mean_reward"])
best_unmasked      = max(unmasked_static, key=lambda r: r["mean_reward"])
best_masked_decay  = max(masked_decay,    key=lambda r: r["mean_reward"])
best_unmasked_decay = max(unmasked_decay, key=lambda r: r["mean_reward"])

np.savez(
    checkpoints_dir / "best_masked_static.npz",
    q_table         = best_masked["q_table"],
    episode_rewards = np.array(best_masked["episode_rewards"]),
    mean_reward     = best_masked["mean_reward"],
    std_reward      = best_masked["std_reward"],
)

np.savez(
    checkpoints_dir / "best_unmasked_static.npz",
    q_table         = best_unmasked["q_table"],
    episode_rewards = np.array(best_unmasked["episode_rewards"]),
    mean_reward     = best_unmasked["mean_reward"],
    std_reward      = best_unmasked["std_reward"],
)

np.savez(
    checkpoints_dir / "best_masked_decay.npz",
    q_table         = best_masked_decay["q_table"],
    episode_rewards = np.array(best_masked_decay["episode_rewards"]),
    mean_reward     = best_masked_decay["mean_reward"],
    std_reward      = best_masked_decay["std_reward"],
)

np.savez(
    checkpoints_dir / "best_unmasked_decay.npz",
    q_table         = best_unmasked_decay["q_table"],
    episode_rewards = np.array(best_unmasked_decay["episode_rewards"]),
    mean_reward     = best_unmasked_decay["mean_reward"],
    std_reward      = best_unmasked_decay["std_reward"],
)

print("\nDone. Files written:")
print(f"  {output_dir}/epsilon_comparison.png")
print(f"  {output_dir}/q_learning_epsilon_schedule_effect.png")
print(f"  {checkpoints_dir}/summary.json")
print(f"  {checkpoints_dir}/best_masked_static.npz")
print(f"  {checkpoints_dir}/best_unmasked_static.npz")
print(f"  {checkpoints_dir}/best_masked_decay.npz")
print(f"  {checkpoints_dir}/best_unmasked_decay.npz")
print(f"  {qtables_dir}/q_learning_masked_static_qtables.npy")
print(f"  {qtables_dir}/q_learning_unmasked_static_qtables.npy")
print(f"  {qtables_dir}/q_learning_masked_decay_qtables.npy")
print(f"  {qtables_dir}/q_learning_unmasked_decay_qtables.npy")
print(f"  {rewards_dir}/masked_static_rewards.npy       # shape: ({n_runs}, {episodes})")
print(f"  {rewards_dir}/unmasked_static_rewards.npy     # shape: ({n_runs}, {episodes})")
print(f"  {rewards_dir}/masked_decay_rewards.npy        # shape: ({n_runs}, {episodes})")
print(f"  {rewards_dir}/unmasked_decay_rewards.npy      # shape: ({n_runs}, {episodes})")