import numpy as np
import imageio
from pathlib import Path
from multi_passenger_taxi import MultiPassengerTaxiEnv, FIXED_LOCS, GRID_ROWS, GRID_COLS, _BLOCKED

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

CELL = 64  # pixels per grid cell — 5×5 grid → 320×320 output

# Colors (RGB)
BG_COLOR    = (30,  30,  30)
GRID_COLOR  = (70,  70,  70)
WALL_COLOR  = (220, 220, 220)
WALL_THICK  = 3
LOC_COLORS  = [          # R, G, Y, B
    (180, 80,  80),
    (80,  180, 80),
    (180, 180, 80),
    (80,  80,  180),
]
TAXI_FREE   = (255, 220, 50)   # yellow — no passenger
TAXI_LOADED = (50,  220, 80)   # green  — passenger aboard
PSGR_COLORS = [(255, 140, 0), (0, 200, 200)]


def _render_frame(env) -> np.ndarray:
    H = GRID_ROWS * CELL
    W = GRID_COLS * CELL
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

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

    # walls — drawn as thick lines on the shared edge between two cells
    # _BLOCKED contains ((row, col), action) pairs; action 2=East means
    # there's a wall on the right edge of (row, col)
    drawn_walls = set()
    for (r, c), action in _BLOCKED:
        if action == 2:   # East wall → right edge of (r, c)
            key = (r, c, "E")
            if key not in drawn_walls:
                x = (c + 1) * CELL
                img[r * CELL: (r + 1) * CELL, x - WALL_THICK: x + WALL_THICK] = WALL_COLOR
                drawn_walls.add(key)
        elif action == 0:  # South wall → bottom edge of (r, c)
            key = (r, c, "S")
            if key not in drawn_walls:
                y = (r + 1) * CELL
                img[y - WALL_THICK: y + WALL_THICK, c * CELL: (c + 1) * CELL] = WALL_COLOR
                drawn_walls.add(key)

    # passengers (draw before taxi so taxi renders on top)
    for p in range(env.n_passengers):
        if env._p_delivered[p]:
            continue
        loc = env._p_loc[p]
        if loc == env.n_locs:  # in taxi
            pr, pc = env._taxi_row, env._taxi_col
        else:
            pr, pc = FIXED_LOCS[loc]
        color  = PSGR_COLORS[p % len(PSGR_COLORS)]
        offset = p * (CELL // 5)
        size   = CELL // 4
        py = pr * CELL + 4 + offset
        px = pc * CELL + 4 + offset
        img[py: py + size, px: px + size] = color

    # taxi
    tr, tc     = env._taxi_row, env._taxi_col
    in_taxi    = any(env._p_loc[p] == env.n_locs for p in range(env.n_passengers))
    taxi_color = TAXI_LOADED if in_taxi else TAXI_FREE
    pad        = CELL // 5
    img[tr * CELL + pad: (tr + 1) * CELL - pad,
        tc * CELL + pad: (tc + 1) * CELL - pad] = taxi_color

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
                valid_actions = np.nonzero(info["action_mask"] == 1)[0]
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
            frames.append(_render_frame(env))  # capture final state
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