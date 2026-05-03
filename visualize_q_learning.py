import numpy as np
import imageio
import gymnasium as gym

BASE_RANDOM_SEED = 58922320
N_RUNS = 12

# Load ALL Q-tables (shape: [12, n_states, n_actions])
all_qtables = np.load("masked_qtables.npy")

def record_episode(q_table, filename="taxi.gif", max_steps=100, epsilon=0.0, seed=None):
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    frames = []

    state, info = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0

    while steps < max_steps:
        frame = env.render()
        frames.append(frame.copy())

        if not (done or truncated):
            action_mask = info["action_mask"]
            valid_actions = np.nonzero(action_mask == 1)[0]

            if np.random.rand() < epsilon or len(valid_actions) == 0:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = valid_actions[np.argmax(q_table[state, valid_actions])]

            state, reward, done, truncated, info = env.step(action)

        steps += 1
        if done or truncated:
            break

    env.close()

    print("Frames collected:", len(frames))
    imageio.mimsave(filename, frames, duration=0.3)
    print(f"Saved {filename}")


# Run all seeds
for i in range(N_RUNS):
    seed = BASE_RANDOM_SEED + i
    run_qtable = all_qtables[i]

    record_episode(
        run_qtable,
        filename=f"./examples/masked_run_{i+1}.gif",
        seed=seed,
        epsilon=0.0,   # deterministic policy for visualization
    )

    print(f"Saved run {i+1}/{N_RUNS}")