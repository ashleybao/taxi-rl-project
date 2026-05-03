"""
multi_passenger_taxi_large.py
==============================
An expanded Gymnasium environment: 15×15 grid, 8 pickup/dropoff stations,
up to 4 passengers, and wall-based obstacles (matching Taxi-v3 style).

Grid layout (15×15):
    Stations (row, col):
        0=NW (0,0)   1=NE (0,14)
        2=MW (4,2)   3=MC (4,7)   4=ME (4,12)
        5=SW (10,2)  6=SC (10,7)  7=SE (14,14)

    Walls trace the perimeter of four city-block regions:
        upper-centre block : rows 1–3,  cols 5–8
        mid-left block     : rows 6–8,  cols 1–4
        mid-right block    : rows 6–8,  cols 10–13
        lower-centre block : rows 11–13, cols 5–8

    The taxi can enter these cells but cannot pass through the perimeter
    walls surrounding each block — matching Taxi-v3's wall convention.

Actions (same 6 as Taxi-v3):
    0 = South (+row)    1 = North (-row)
    2 = East  (+col)    3 = West  (-col)
    4 = Pickup          5 = Dropoff

Rewards:
    -1  : time penalty per step
    +20 : per successful delivery
    -10 : illegal pickup / dropoff

Action mask: int8 array of length 6 (1=valid).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

GRID_ROWS = 15
GRID_COLS = 15

FIXED_LOCS: list[tuple[int, int]] = [
    (0,  0),   # 0 NW
    (0, 14),   # 1 NE
    (4,  2),   # 2 MW
    (4,  7),   # 3 MC
    (4, 12),   # 4 ME
    (10, 2),   # 5 SW
    (10, 7),   # 6 SC
    (14,14),   # 7 SE
]

LOC_LABELS = ["NW", "NE", "MW", "MC", "ME", "SW", "SC", "SE"]

# No obstacle cells — the grid is fully traversable, walls limit movement
OBSTACLE_CELLS: frozenset[tuple[int, int]] = frozenset()

# ---------------------------------------------------------------------------
# Wall layout
# Each city-block region has walls along its perimeter edges.
# _BLOCKED stores (cell, action) pairs that are forbidden:
#   action 0=South, 1=North, 2=East, 3=West
# ---------------------------------------------------------------------------

_WALL_BLOCKS: list[tuple[int, int, int, int]] = [
    # (row_start, row_end_inclusive, col_start, col_end_inclusive)
    (1,  3,  5,  8),   # upper-centre
    (6,  8,  1,  4),   # mid-left
    (6,  8, 10, 13),   # mid-right
    (11, 13, 5,  8),   # lower-centre
]

_BLOCKED: set[tuple[tuple[int, int], int]] = set()

for (_r0, _r1, _c0, _c1) in _WALL_BLOCKS:
    # Top edge: wall between row r0-1 and r0
    if _r0 > 0:
        for _c in range(_c0, _c1 + 1):
            _BLOCKED.add(((_r0 - 1, _c), 0))   # South from above
            _BLOCKED.add(((_r0,     _c), 1))   # North from inside
    # Bottom edge: wall between row r1 and r1+1
    if _r1 < GRID_ROWS - 1:
        for _c in range(_c0, _c1 + 1):
            _BLOCKED.add(((_r1,     _c), 0))   # South from inside
            _BLOCKED.add(((_r1 + 1, _c), 1))   # North from below
    # Left edge: wall between col c0-1 and c0
    if _c0 > 0:
        for _r in range(_r0, _r1 + 1):
            _BLOCKED.add(((_r, _c0 - 1), 2))   # East from left
            _BLOCKED.add(((_r, _c0),     3))   # West from inside
    # Right edge: wall between col c1 and c1+1
    if _c1 < GRID_COLS - 1:
        for _r in range(_r0, _r1 + 1):
            _BLOCKED.add(((_r, _c1),     2))   # East from inside
            _BLOCKED.add(((_r, _c1 + 1), 3))   # West from right

# Sanity-check: all stations are reachable (not walled off on all sides)
for _loc in FIXED_LOCS:
    assert _loc not in OBSTACLE_CELLS, f"Station {_loc} inside obstacle!"

# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

_DELTAS = {
    0: (1,  0),   # South
    1: (-1, 0),   # North
    2: (0,  1),   # East
    3: (0, -1),   # West
}


def _move(row: int, col: int, action: int) -> tuple[int, int]:
    """Return new (row, col) after action, respecting boundaries and walls."""
    if action not in _DELTAS:
        return row, col
    dr, dc = _DELTAS[action]
    nr, nc = row + dr, col + dc
    if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
        return row, col
    if ((row, col), action) in _BLOCKED:
        return row, col
    return nr, nc


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LargeMultiPassengerTaxiEnv(gym.Env):
    """
    15×15 multi-passenger taxi with wall-based city-block obstacles.

    Parameters
    ----------
    n_passengers : int  (1–4)
    max_steps    : int  episode truncation limit
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, n_passengers: int = 2, max_steps: int = 500) -> None:
        super().__init__()

        if not (1 <= n_passengers <= 4):
            raise ValueError("n_passengers must be 1–4.")
        if n_passengers > len(FIXED_LOCS):
            raise ValueError("n_passengers exceeds available stations.")

        self.n_passengers = n_passengers
        self.max_steps    = max_steps
        self.n_locs       = len(FIXED_LOCS)   # 8
        self._p_states    = self.n_locs + 1   # 9  (0..7=waiting, 8=in-taxi)

        n_states = (
            GRID_ROWS * GRID_COLS
            * (self._p_states ** n_passengers)
            * (self.n_locs    ** n_passengers)
        )

        self.observation_space = spaces.Discrete(n_states)
        self.action_space      = spaces.Discrete(6)

        self._taxi_row:    int        = 0
        self._taxi_col:    int        = 0
        self._p_loc:       list[int]  = [0] * n_passengers
        self._p_dest:      list[int]  = [0] * n_passengers
        self._p_delivered: list[bool] = [False] * n_passengers
        self._steps:       int        = 0

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _encode(self) -> int:
        s = self._taxi_row * GRID_COLS + self._taxi_col
        for p in range(self.n_passengers):
            s = s * self._p_states + self._p_loc[p]
        for p in range(self.n_passengers):
            s = s * self.n_locs + self._p_dest[p]
        return int(s)

    def _decode(self, state: int) -> tuple[int, int, list[int], list[int]]:
        p_dests = []
        for _ in range(self.n_passengers):
            p_dests.append(state % self.n_locs)
            state //= self.n_locs
        p_dests.reverse()

        p_locs = []
        for _ in range(self.n_passengers):
            p_locs.append(state % self._p_states)
            state //= self._p_states
        p_locs.reverse()

        taxi_row = state // GRID_COLS
        taxi_col = state % GRID_COLS
        return taxi_row, taxi_col, p_locs, p_dests

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _taxi_loc_index(self) -> int | None:
        cell = (self._taxi_row, self._taxi_col)
        try:
            return FIXED_LOCS.index(cell)
        except ValueError:
            return None

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.ones(6, dtype=np.int8)
        row, col = self._taxi_row, self._taxi_col

        for a in range(4):
            nr, nc = _move(row, col, a)
            if nr == row and nc == col:
                mask[a] = 0

        taxi_loc = self._taxi_loc_index()
        in_taxi  = [p for p in range(self.n_passengers) if self._p_loc[p] == self.n_locs]

        can_pickup = (
            taxi_loc is not None
            and len(in_taxi) == 0
            and any(
                self._p_loc[p] == taxi_loc and not self._p_delivered[p]
                for p in range(self.n_passengers)
            )
        )
        mask[4] = int(can_pickup)

        can_dropoff = (
            len(in_taxi) > 0
            and taxi_loc is not None
            and any(self._p_dest[p] == taxi_loc for p in in_taxi)
        )
        mask[5] = int(can_dropoff)

        if mask.sum() == 0:
            mask[:4] = 1
        return mask

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Taxi starts at any cell (all cells reachable with wall-based obstacles)
        self._taxi_row = int(rng.integers(0, GRID_ROWS))
        self._taxi_col = int(rng.integers(0, GRID_COLS))

        loc_indices = list(range(self.n_locs))
        start_locs  = rng.choice(loc_indices, size=self.n_passengers, replace=False).tolist()

        for p in range(self.n_passengers):
            self._p_loc[p]  = int(start_locs[p])
            possible_dests  = [l for l in loc_indices if l != start_locs[p]]
            self._p_dest[p] = int(rng.choice(possible_dests))

        self._steps       = 0
        self._p_delivered = [False] * self.n_passengers

        obs  = self._encode()
        info = {"action_mask": self._compute_action_mask()}
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        self._steps += 1
        reward = -1.0
        done   = False

        if action < 4:
            self._taxi_row, self._taxi_col = _move(
                self._taxi_row, self._taxi_col, action
            )

        elif action == 4:  # Pickup
            taxi_loc = self._taxi_loc_index()
            in_taxi  = [p for p in range(self.n_passengers) if self._p_loc[p] == self.n_locs]
            if taxi_loc is not None and len(in_taxi) == 0:
                picked = False
                for p in range(self.n_passengers):
                    if self._p_loc[p] == taxi_loc and not self._p_delivered[p]:
                        self._p_loc[p] = self.n_locs
                        picked = True
                        break
                if not picked:
                    reward = -10.0
            else:
                reward = -10.0

        elif action == 5:  # Dropoff
            taxi_loc = self._taxi_loc_index()
            in_taxi  = [p for p in range(self.n_passengers) if self._p_loc[p] == self.n_locs]
            if in_taxi and taxi_loc is not None:
                dropped = False
                for p in in_taxi:
                    if self._p_dest[p] == taxi_loc:
                        self._p_loc[p]       = taxi_loc
                        self._p_delivered[p] = True
                        reward += 20.0
                        dropped = True
                        break
                if not dropped:
                    reward = -10.0
            else:
                reward = -10.0

        done      = all(self._p_delivered)
        truncated = (self._steps >= self.max_steps) and not done
        obs       = self._encode()
        info      = {"action_mask": self._compute_action_mask()}
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Feature vector for function approximation
    # ------------------------------------------------------------------

    def feature_vector(self, obs: int | None = None) -> np.ndarray:
        if obs is not None:
            tr, tc, p_locs, p_dests = self._decode(obs)
            delivered = [p_locs[p] == p_dests[p] and p_locs[p] != self.n_locs
                         for p in range(self.n_passengers)]
            if obs == self._encode():
                delivered = list(self._p_delivered)
        else:
            tr, tc    = self._taxi_row, self._taxi_col
            p_locs    = self._p_loc
            p_dests   = self._p_dest
            delivered = self._p_delivered

        P = self.n_passengers
        L = self.n_locs
        max_dist = float(GRID_ROWS + GRID_COLS - 2)

        parts = []
        parts.append(np.array([tr / (GRID_ROWS - 1), tc / (GRID_COLS - 1)], dtype=np.float32))

        for p in range(P):
            oh = np.zeros(L + 1, dtype=np.float32)
            oh[p_locs[p]] = 1.0
            parts.append(oh)

        for p in range(P):
            oh = np.zeros(L, dtype=np.float32)
            oh[p_dests[p]] = 1.0
            parts.append(oh)

        parts.append(np.array([float(d) for d in delivered], dtype=np.float32))

        waiting_dists, intaxi_dists = [], []
        for p in range(P):
            if not delivered[p]:
                if p_locs[p] < L:
                    lr, lc = FIXED_LOCS[p_locs[p]]
                    waiting_dists.append(abs(tr - lr) + abs(tc - lc))
                else:
                    dr2, dc2 = FIXED_LOCS[p_dests[p]]
                    intaxi_dists.append(abs(tr - dr2) + abs(tc - dc2))

        nearest_waiting = min(waiting_dists) / max_dist if waiting_dists else 0.0
        nearest_intaxi  = min(intaxi_dists)  / max_dist if intaxi_dists  else 0.0
        parts.append(np.array([nearest_waiting, nearest_intaxi], dtype=np.float32))

        return np.concatenate(parts)

    @property
    def feature_dim(self) -> int:
        P, L = self.n_passengers, self.n_locs
        return 2 + P * (L + 1) + P * L + P + 2

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        loc_ch = {loc: lbl[0] for loc, lbl in zip(FIXED_LOCS, LOC_LABELS)}
        rows_out = []
        border = "+" + "-" * (GRID_COLS * 2 - 1) + "+"
        rows_out.append(border)
        for r in range(GRID_ROWS):
            row_str = "|"
            for c in range(GRID_COLS):
                if (r, c) == (self._taxi_row, self._taxi_col):
                    ch = "T"
                elif (r, c) in loc_ch:
                    ch = loc_ch[(r, c)]
                else:
                    ch = "."
                # right wall
                if c < GRID_COLS - 1 and ((r, c), 2) in _BLOCKED:
                    sep = "|"
                else:
                    sep = " "
                row_str += ch + sep
            rows_out.append(row_str.rstrip() + "|")
            # horizontal wall row below
            if r < GRID_ROWS - 1:
                wall_row = " "
                for c in range(GRID_COLS):
                    wall_row += "-" if ((r, c), 0) in _BLOCKED else " "
                    wall_row += " "
                rows_out.append(wall_row)
        rows_out.append(border)

        for p in range(self.n_passengers):
            loc  = self._p_loc[p]
            dest = self._p_dest[p]
            loc_str = "in-taxi" if loc == self.n_locs else f"{LOC_LABELS[loc]}{FIXED_LOCS[loc]}"
            rows_out.append(
                f"  P{p}: {loc_str} → {LOC_LABELS[dest]}{FIXED_LOCS[dest]}"
                + (" ✓" if self._p_delivered[p] else "")
            )
        rows_out.append(f"  Steps: {self._steps}/{self.max_steps}")
        result = "\n".join(rows_out)
        print(result)
        return result

    def get_map_info(self) -> dict:
        return {
            "grid_rows":    GRID_ROWS,
            "grid_cols":    GRID_COLS,
            "fixed_locs":   FIXED_LOCS,
            "loc_labels":   LOC_LABELS,
            "wall_blocks":  _WALL_BLOCKS,
            "n_blocked":    len(_BLOCKED),
            "n_passengers": self.n_passengers,
            "feature_dim":  self.feature_dim,
            "obs_space":    self.observation_space.n,
        }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = LargeMultiPassengerTaxiEnv(n_passengers=2, max_steps=500)
    obs, info = env.reset(seed=0)
    info2 = env.get_map_info()

    print("=== LargeMultiPassengerTaxiEnv (wall-based) ===")
    print(f"Grid          : {GRID_ROWS}×{GRID_COLS}")
    print(f"Wall segments : {info2['n_blocked'] // 2} unique edges")
    print(f"Stations      : {len(FIXED_LOCS)}")
    print(f"Obs space     : {info2['obs_space']:,}")
    print(f"Feature dim   : {info2['feature_dim']}")
    print()
    env.render()

    total_reward = 0.0
    for _ in range(50):
        valid  = np.nonzero(info["action_mask"])[0]
        action = int(np.random.choice(valid))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"\nReward after up to 50 random steps: {total_reward}")
    print("Smoke-test passed.")