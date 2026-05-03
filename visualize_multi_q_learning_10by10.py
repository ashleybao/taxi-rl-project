import numpy as np
import imageio
from pathlib import Path
from multi_passenger_taxi import MultiPassengerTaxiEnv, FIXED_LOCS, GRID_ROWS, GRID_COLS, OBSTACLE_CELLS

BASE_RANDOM_SEED = 58922320
N_RUNS           = 12
N_PASSENGERS     = 2

qtables_dir  = Path("results/q_learning/multi_passenger/q_tables")
examples_dir = Path("results/q_learning/multi_passenger/examples")
examples_dir.mkdir(parents=True, exist_ok=True)

conditions = {
    "masked_static":   qtables_dir / "multi_masked_static_qtables.npy",
    "unmasked_static": qtables_dir / "multi_unmasked_static_qtables.npy",
    "masked_decay":    qtables_dir / "multi_masked_decay_qtables.npy",
    "unmasked_decay":  qtables_dir / "multi_unmasked_decay_qtables.npy",
}

CELL = 32  # pixels per grid cell

# Colors (RGB)
BG_COLOR    = (30,  30,  30)
GRID_COLOR  = (70,  70,  70)
OBS_COLOR   = (50,  50,  50)    # dark — obstacle blocks
LOC_COLORS  = [                  # one tint per station (up to 4 stations)
    (180, 80,  80),
    (80,  180, 80),
    (80,  80,  180),
    (180, 180, 80),
]
TAXI_FREE   = (255, 220, 50)    # yellow
TAXI_LOADED = (50,  220, 80)    # green
PSGR_COLORS = [(255, 140, 0), (0, 200, 200)]  # one per passenger


def _render_frame(env) -> np.ndarray:
    H = GRID_ROWS * CELL
    W = GRID_COLS * CELL
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    # obstacle cells
    for (r, c) in OBSTACLE_CELLS:
        y, x = r * CELL, c * CELL
        img[y:y + CELL, x:x + CELL] = OBS_COLOR

    # station location tints
    for idx, (r, c) in enumerate(FIXED_LOCS):
        color = LOC_COLORS[idx % len(LOC_COLORS)]
        y, x = r * CELL + 1, c * CELL + 1
        img[y:y + CELL - 2, x:x + CELL - 2] = color

    # grid lines
    for i in range(GRID_ROWS + 1):
        img[i * CELL: i * CELL + 1, :] = GRID_COLOR
    for j in range(GRID_COLS + 1):
        img[:, j * CELL: j * CELL + 1] = GRID_COLOR

    # passengers (draw before taxi so taxi renders on top)
    for p in range(env.n_passengers):
        if env._p_delivered[p]:
            continue
        loc = env._p_loc[p]
        if loc == env.n_locs:  # in taxi — draw as small dot on taxi cell
            pr, pc = env._taxi_row, env._taxi_col
        else:
            pr, pc = FIXED_LOCS[loc]
        color  = PSGR_COLORS[p % len(PSGR_COLORS)]
        offset = p * (CELL // 4)
        py = pr * CELL + 2 + offset
        px = pc * CELL + 2 + offset
        size = CELL // 4
        img[py: py + size, px: px + size] = color

    # taxi
    tr, tc      = env._taxi_row, env._taxi_col
    in_taxi     = any(env._p_loc[p] == env.n_locs for p in range(env.n_passengers))
    taxi_color  = TAXI_LOADED if in_taxi else TAXI_FREE
    pad         = CELL // 5
    ty, tx      = tr * CELL + pad, tc * CELL + pad
    img[ty: ty + CELL - 2 * pad, tx: tx + CELL - 2 * pad] = taxi_color

    return img


def record_episode(
    q_table,
    filename,
    use_action_mask=True,
    max_steps=200,
    epsilon=0.0,
    seed=None,
):
    env = MultiPassengerTaxiEnv(n_passengers=N_PASSENGERS, max_steps=max_steps)
    frames = []
    state, info = env.reset(seed=seed)
    done = truncated = False
    steps = 0

    while steps < max_steps:
        frames.append(_render_frame(env))

        if not (done or truncated):
            if use_action_mask:
                action_mask   = info["action_mask"]
                valid_actions = np.nonzero(action_mask == 1)[0]
                if np.random.rand() < epsilon or len(valid_actions) == 0:
                    action = int(np.random.randint(0, env.action_space.n))
                else:
                    action = int(valid_actions[np.argmax(q_table[state, valid_actions])])
            else:
                if np.random.rand() < epsilon:
                    action = int(np.random.randint(0, env.action_space.n))
                else:
                    action = int(np.argmax(q_table[state]))

            state, reward, done, truncated, info = env.step(action)

        steps += 1
        if done or truncated:
            frames.append(_render_frame(env))  # final frame
            break

    env.close()
    imageio.mimsave(str(filename), frames, duration=0.2)
    print(f"  Saved {filename}  ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Record all runs for all conditions
# ---------------------------------------------------------------------------

for condition, qtable_path in conditions.items():
    print(f"\n=== {condition} ===")
    all_qtables = np.load(str(qtable_path))
    use_mask    = "unmasked" not in condition

    out_dir = examples_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_RUNS):
        seed = BASE_RANDOM_SEED + i
        record_episode(
            all_qtables[i],
            filename        = out_dir / f"run_{i+1}.gif",
            use_action_mask = use_mask,
            seed            = seed,
            epsilon         = 0.0,
        )
        print(f"  Run {i+1}/{N_RUNS} done")