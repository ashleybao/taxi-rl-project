import numpy as np
import imageio
from pathlib import Path
from multi_passenger_taxi import MultiPassengerTaxiEnv, FIXED_LOCS

# ---------------------------------------------------------------------
# Paths (SARSA outputs)
# ---------------------------------------------------------------------
RESULTS_DIR = Path("results/sarsa/multi")

ALL_QTABLES_PATH = RESULTS_DIR / "multi_masked_qtables.npy"
BEST_RUN_PATH    = RESULTS_DIR / "best_multi_masked_run.npz"

# ---------------------------------------------------------------------
# Load Q-tables
# ---------------------------------------------------------------------
data = np.load(BEST_RUN_PATH, allow_pickle=True)
best_q_table = data["q_table"]

all_qtables = np.load(ALL_QTABLES_PATH, allow_pickle=True)  # (12, states, actions)

# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def record_episode(q_table, filename="taxi_multi.gif", max_steps=200, epsilon=0.0, seed=None):
    env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=max_steps)

    frames = []
    state, info = env.reset(seed=seed)
    done = False
    truncated = False
    steps = 0

    while steps < max_steps:
        frames.append(_render_frame(env))

        if not (done or truncated):
            action_mask = info["action_mask"]
            valid_actions = np.nonzero(action_mask == 1)[0]

            if np.random.rand() < epsilon or len(valid_actions) == 0:
                action = int(np.random.randint(0, env.action_space.n))
            else:
                action = int(valid_actions[np.argmax(q_table[state, valid_actions])])

            state, reward, done, truncated, info = env.step(action)

        steps += 1
        if done or truncated:
            break

    env.close()

    imageio.mimsave(filename, frames, fps=3)
    print(f"Saved {filename} ({len(frames)} frames)")


# ---------------------------------------------------------------------
# Grid rendering (unchanged)
# ---------------------------------------------------------------------
def _render_frame(env) -> np.ndarray:
    CELL = 60
    H = W = 5 * CELL
    img = np.full((H, W, 3), 30, dtype=np.uint8)

    GRID_COLOR  = (80, 80, 80)
    WALL_COLOR  = (220, 220, 220)
    WALL_THICK  = 4
    TAXI_COLOR  = (255, 220, 50)
    LOC_COLORS  = [(200, 80, 80), (80, 200, 80), (80, 80, 200), (200, 200, 80)]
    PSGR_COLORS = [(255, 140, 0), (0, 200, 200)]

    for i in range(5):
        img[i * CELL, :] = GRID_COLOR
        img[:, i * CELL] = GRID_COLOR

    for idx, (r, c) in enumerate(FIXED_LOCS):
        y, x = r * CELL + 4, c * CELL + 4
        img[y:y + CELL - 8, x:x + CELL - 8] = LOC_COLORS[idx]

    tr, tc = env._taxi_row, env._taxi_col
    ty, tx = tr * CELL + 12, tc * CELL + 12
    img[ty:ty + CELL - 24, tx:tx + CELL - 24] = TAXI_COLOR

    for p in range(env.n_passengers):
        loc = env._p_loc[p]
        if env._p_delivered[p]:
            continue

        pr, pc = (tr, tc) if loc == env.n_locs else FIXED_LOCS[loc]
        offset = p * 12
        py, px = pr * CELL + 2 + offset, pc * CELL + 2 + offset
        img[py:py + 10, px:px + 10] = PSGR_COLORS[p % len(PSGR_COLORS)]

    walls = [
        (0, 1), (1, 1),
        (3, 0), (4, 0),
        (3, 2), (4, 2),
    ]

    for (r, c) in walls:
        x = (c + 1) * CELL
        y_start = r * CELL
        y_end = (r + 1) * CELL
        img[y_start:y_end, x - WALL_THICK // 2 : x + WALL_THICK // 2] = WALL_COLOR

    return img


# ---------------------------------------------------------------------
# Generate GIFs for all SARSA runs
# ---------------------------------------------------------------------
BASE_RANDOM_SEED = 58922320
N_RUNS = 12

for i in range(N_RUNS):
    seed = BASE_RANDOM_SEED + i
    run_qtable = all_qtables[i]

    record_episode(
        run_qtable,
        filename=RESULTS_DIR / "gifs" / f"multi_sarsa_masked_run_{i+1}.gif",
        seed=seed,
        epsilon=0.0,
    )

    print(f"Saved SARSA run {i+1}/{N_RUNS}")