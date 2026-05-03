import numpy as np
import imageio
from pathlib import Path
from bigger_taxi_env import BiggerTaxiEnv

BASE_RANDOM_SEED = 58922320
N_RUNS = 12

qtables_dir = Path("results/10by10map/q_learning/q_tables")
examples_dir = Path("results/10by10map/q_learning/examples")
examples_dir.mkdir(parents=True, exist_ok=True)

# Load all Q-tables for each condition (shape: [12, n_states, n_actions])
conditions = {
    "masked_static":   qtables_dir / "q_learning_masked_static_qtables.npy",
    "unmasked_static": qtables_dir / "q_learning_unmasked_static_qtables.npy",
    "masked_decay":    qtables_dir / "q_learning_masked_decay_qtables.npy",
    "unmasked_decay":  qtables_dir / "q_learning_unmasked_decay_qtables.npy",
}


def record_episode(q_table, filename, use_action_mask=True, max_steps=200, epsilon=0.0, seed=None):
    env = BiggerTaxiEnv(size=10)
    frames = []
    state, info = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0

    while steps < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())

        if not (done or truncated):
            if use_action_mask:
                action_mask = info["action_mask"]
                valid_actions = np.nonzero(action_mask == 1)[0]
                if np.random.rand() < epsilon or len(valid_actions) == 0:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    action = valid_actions[np.argmax(q_table[state, valid_actions])]
            else:
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    action = np.argmax(q_table[state])

            state, reward, done, truncated, info = env.step(action)

        steps += 1
        if done or truncated:
            # capture final frame
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            break

    env.close()
    print(f"  Frames collected: {len(frames)}")
    imageio.mimsave(filename, frames, duration=0.3)
    print(f"  Saved {filename}")


# Record all runs for all conditions
for condition, qtable_path in conditions.items():
    print(f"\n=== {condition} ===")
    all_qtables = np.load(qtable_path)
    use_mask = "unmasked" not in condition

    for i in range(N_RUNS):
        seed = BASE_RANDOM_SEED + i
        out_path = examples_dir / condition / f"run_{i+1}.gif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        record_episode(
            all_qtables[i],
            filename=str(out_path),
            use_action_mask=use_mask,
            seed=seed,
            epsilon=0.0,  # deterministic policy for visualization
        )
        print(f"  Run {i+1}/{N_RUNS} done")