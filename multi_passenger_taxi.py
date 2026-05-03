"""
multi_passenger_taxi.py
=======================
A custom Gymnasium environment that extends the classic Taxi-v3 grid to support
multiple passengers simultaneously.

Grid layout (5×5, same as Taxi-v3):
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

Fixed pickup/dropoff locations (row, col):
    R = (0, 0)
    G = (0, 4)
    Y = (4, 0)
    B = (4, 3)

Actions (same 6 as Taxi-v3):
    0 = South   (+row)
    1 = North   (-row)
    2 = East    (+col)
    3 = West    (-col)
    4 = Pickup
    5 = Dropoff

State encoding:
    Each state is a single integer that encodes:
        taxi_row, taxi_col, p0_loc, p1_loc, ..., p(n-1)_loc

    Each passenger location is one of (n_locs + 1) values:
        0..n_locs-1  → waiting at that fixed location
        n_locs       → currently in the taxi

    Total states = 25 * (n_locs + 1)^n_passengers

Rewards (per step):
    -1   : every step (time penalty)
    +20  : per passenger successfully delivered
    -10  : illegal pickup or dropoff attempt

Action mask:
    A numpy int8 array of length 6.
    1 = valid action, 0 = invalid action.

    Move actions are invalid when a wall blocks the path.
    Pickup is invalid when the taxi is NOT on a waiting passenger.
    Dropoff is invalid when:
        - no passenger is in the taxi, OR
        - the taxi is not at a correct destination for any in-taxi passenger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Grid constants  (identical to Taxi-v3)
# ---------------------------------------------------------------------------

GRID_ROWS = 5
GRID_COLS = 5

# Fixed locations: R, G, Y, B
FIXED_LOCS: list[tuple[int, int]] = [
    (0, 0),  # R
    (0, 4),  # G
    (4, 0),  # Y
    (4, 3),  # B
]

# Walls stored as frozensets of ((r,c), (r,c)) pairs that cannot be crossed.
# Each entry blocks movement between the two adjacent cells.
# Derived from the Taxi-v3 source map.
_WALLS: frozenset[frozenset] = frozenset(
    frozenset(pair)
    for pair in [
        # vertical walls (block East/West movement)
        ((0, 1), (0, 2)),
        ((1, 1), (1, 2)),
        ((0, 0), (0, 1)),   # not a real wall — placeholder guard removed below
        ((3, 0), (3, 1)),
        ((4, 0), (4, 1)),
        ((3, 2), (3, 3)),
        ((4, 2), (4, 3)),
    ]
)

# It is cleaner to store walls as a set of (from_cell, action) pairs.
# Build them explicitly so they are easy to audit.
#
# Action encoding:  0=S 1=N 2=E 3=W
# Wall between cell A and cell B (A→B blocked, B→A blocked):
#   If B is East  of A: block (A, East=2) and (B, West=3)
#   If B is South of A: block (A, South=0) and (B, North=1)

_BLOCKED: set[tuple[tuple[int, int], int]] = set()


def _add_wall(r1: int, c1: int, r2: int, c2: int) -> None:
    """Register a wall between adjacent cells (r1,c1)↔(r2,c2)."""
    if r2 == r1 and c2 == c1 + 1:        # (r1,c1) → East → (r1,c2)
        _BLOCKED.add(((r1, c1), 2))       # East from left cell
        _BLOCKED.add(((r1, c2), 3))       # West from right cell
    elif r2 == r1 + 1 and c2 == c1:      # (r1,c1) → South → (r2,c1)
        _BLOCKED.add(((r1, c1), 0))       # South from top cell
        _BLOCKED.add(((r2, c1), 1))       # North from bottom cell
    else:
        raise ValueError(f"Cells ({r1},{c1})↔({r2},{c2}) are not adjacent.")


# Taxi-v3 walls (vertical dividers visible in the grid art)
_add_wall(0, 1, 0, 2)
_add_wall(1, 1, 1, 2)
_add_wall(3, 0, 3, 1)
_add_wall(4, 0, 4, 1)
_add_wall(3, 2, 3, 3)
_add_wall(4, 2, 4, 3)


# ---------------------------------------------------------------------------
# Helper: clamp movement to grid boundaries
# ---------------------------------------------------------------------------

_DELTAS = {
    0: (1, 0),   # South
    1: (-1, 0),  # North
    2: (0, 1),   # East
    3: (0, -1),  # West
}


def _move(row: int, col: int, action: int) -> tuple[int, int]:
    """Return the new (row, col) after attempting a move, respecting walls & bounds."""
    if action not in _DELTAS:
        return row, col
    dr, dc = _DELTAS[action]
    new_row = row + dr
    new_col = col + dc
    # Out-of-bounds → stay
    if not (0 <= new_row < GRID_ROWS and 0 <= new_col < GRID_COLS):
        return row, col
    # Wall → stay
    if ((row, col), action) in _BLOCKED:
        return row, col
    return new_row, new_col


# ---------------------------------------------------------------------------
# MultiPassengerTaxiEnv
# ---------------------------------------------------------------------------

class MultiPassengerTaxiEnv(gym.Env):
    """Multi-passenger extension of Taxi-v3.

    Parameters
    ----------
    n_passengers : int
        Number of passengers (≥ 1).  Each passenger starts at a randomly
        chosen fixed location and must be delivered to a *different* fixed
        location.
    max_steps : int
        Episode is truncated after this many steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_passengers: int = 2, max_steps: int = 200) -> None:
        super().__init__()

        if n_passengers < 1:
            raise ValueError("n_passengers must be ≥ 1.")
        if n_passengers > len(FIXED_LOCS):
            raise ValueError(
                f"n_passengers ({n_passengers}) exceeds available "
                f"fixed locations ({len(FIXED_LOCS)})."
            )

        self.n_passengers = n_passengers
        self.max_steps    = max_steps
        self.n_locs       = len(FIXED_LOCS)          # 4

        # Number of passenger-location values: 0..n_locs-1 = waiting, n_locs = in taxi
        self._p_states = self.n_locs + 1             # 5

        # Total discrete states
        n_states = (GRID_ROWS * GRID_COLS 
            * (self._p_states ** self.n_passengers)      # locations
            * (self.n_locs ** self.n_passengers))         # destinations

        self.observation_space = spaces.Discrete(n_states)
        self.action_space      = spaces.Discrete(6)   # S N E W pickup dropoff

        # Internal state (set by reset)
        self._taxi_row:   int       = 0
        self._taxi_col:   int       = 0
        self._p_loc:      list[int] = [0] * n_passengers   # loc index or n_locs (in taxi)
        self._p_dest:     list[int] = [0] * n_passengers   # destination loc index
        self._steps:      int       = 0
        self._p_delivered: list[bool] = [False] * n_passengers

    # ------------------------------------------------------------------
    # State encoding / decoding
    # ------------------------------------------------------------------

    def _encode(self) -> int:
        s = self._taxi_row * GRID_COLS + self._taxi_col
        for p in range(self.n_passengers):
            s = s * self._p_states + self._p_loc[p]
        for p in range(self.n_passengers):          # append destinations
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

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.ones(6, dtype=np.int8)
        row, col = self._taxi_row, self._taxi_col

        # Movement: only block actual walls/boundaries
        for action in range(4):
            new_row, new_col = _move(row, col, action)
            if new_row == row and new_col == col:
                mask[action] = 0

        taxi_loc = self._taxi_cell_loc_index()
        in_taxi  = [p for p in range(self.n_passengers)
                    if self._p_loc[p] == self.n_locs]

        # Pickup: valid if on a waiting passenger AND hands are empty
        can_pickup = (
            taxi_loc is not None
            and len(in_taxi) == 0
            and any(
                self._p_loc[p] == taxi_loc and not self._p_delivered[p]   # ADD delivered check
                for p in range(self.n_passengers)
            )
        )
        mask[4] = int(can_pickup)

        # Dropoff: valid if carrying someone AND at their destination
        can_dropoff = (
            len(in_taxi) > 0
            and taxi_loc is not None
            and any(self._p_dest[p] == taxi_loc for p in in_taxi)
        )
        mask[5] = int(can_dropoff)

        # Safety: guarantee at least one action is always valid (the 4 moves cover this
        # in practice, but guard against edge cases)
        if mask.sum() == 0:
            mask[:4] = 1

        return mask

    def _taxi_cell_loc_index(self) -> int | None:
        """Return the FIXED_LOCS index of the taxi's current cell, or None."""
        cell = (self._taxi_row, self._taxi_col)
        try:
            return FIXED_LOCS.index(cell)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict]:
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        # Random taxi position
        self._taxi_row = int(rng.integers(0, GRID_ROWS))
        self._taxi_col = int(rng.integers(0, GRID_COLS))

        # Each passenger gets a unique starting location and a different destination
        loc_indices   = list(range(self.n_locs))
        start_locs    = rng.choice(loc_indices, size=self.n_passengers, replace=False).tolist()

        for p in range(self.n_passengers):
            self._p_loc[p] = int(start_locs[p])
            # Destination must differ from starting location
            possible_dests = [l for l in loc_indices if l != start_locs[p]]
            self._p_dest[p] = int(rng.choice(possible_dests))

        self._steps = 0
        self._p_delivered = [False] * self.n_passengers

        obs  = self._encode()
        info = {"action_mask": self._compute_action_mask()}
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        self._steps += 1
        reward  = -1.0
        done    = False

        if action < 4:
            # Movement
            self._taxi_row, self._taxi_col = _move(
                self._taxi_row, self._taxi_col, action
            )

        elif action == 4:
            taxi_loc = self._taxi_cell_loc_index()
            in_taxi  = [p for p in range(self.n_passengers)
                        if self._p_loc[p] == self.n_locs]

            if taxi_loc is not None and len(in_taxi) == 0:
                picked = False
                for p in range(self.n_passengers):
                    if self._p_loc[p] == taxi_loc and not self._p_delivered[p]:  # ADD delivered check
                        self._p_loc[p] = self.n_locs
                        picked = True
                        break
                if not picked:
                    reward = -10.0
            else:
                reward = -10.0

        elif action == 5:
            taxi_loc = self._taxi_cell_loc_index()
            in_taxi  = [p for p in range(self.n_passengers)
                        if self._p_loc[p] == self.n_locs]

            if in_taxi and taxi_loc is not None:
                dropped = False
                for p in in_taxi:
                    if self._p_dest[p] == taxi_loc:
                        self._p_loc[p] = taxi_loc
                        self._p_delivered[p] = True    # MARK AS DELIVERED
                        reward += 20.0
                        dropped = True
                        break
                if not dropped:
                    reward = -10.0
            else:
                reward = -10.0

        # Episode ends when all passengers are delivered to their destinations
        done = all(self._p_delivered[p] for p in range(self.n_passengers))

        truncated = (self._steps >= self.max_steps) and not done
        obs       = self._encode()
        info      = {"action_mask": self._compute_action_mask()}

        return obs, reward, done, truncated, info

    def render(self) -> None:
        """Simple ASCII render for debugging."""
        grid = [["." for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        loc_labels = ["R", "G", "Y", "B"]

        for idx, (r, c) in enumerate(FIXED_LOCS):
            grid[r][c] = loc_labels[idx]

        tr, tc = self._taxi_row, self._taxi_col
        grid[tr][tc] = "T"

        print("+-------+")
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("+-------+")

        for p in range(self.n_passengers):
            loc = self._p_loc[p]
            dest = self._p_dest[p]
            if loc == self.n_locs:
                loc_str = "in taxi"
            else:
                loc_str = f"{loc_labels[loc]} {FIXED_LOCS[loc]}"
            print(
                f"  Passenger {p}: at {loc_str} → dest {loc_labels[dest]} {FIXED_LOCS[dest]}"
            )
        print(f"  Steps: {self._steps}/{self.max_steps}\n")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    obs, info = env.reset(seed=42)
    print(f"obs_space size : {env.observation_space.n}")
    print(f"Initial obs    : {obs}")
    print(f"Action mask    : {info['action_mask']}")
    env.render()

    total_reward = 0.0
    for _ in range(20):
        valid = np.nonzero(info["action_mask"])[0]
        action = int(np.random.choice(valid))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"Reward after up to 20 steps: {total_reward}")
    env.close()
    print("Smoke-test passed.")