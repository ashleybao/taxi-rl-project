"""
taxi_v3_wrapper.py
==================
A thin wrapper around Gymnasium's Taxi-v3 that exposes the same internal
state attributes as MultiPassengerTaxiEnv and SinglePassengerLargeTaxiEnv,
so that linear_fa.py's features_from_env() works unchanged.

Taxi-v3 state encoding (from Gymnasium source):
    state = row*100 + col*20 + pass_loc*4 + dest
    - row:      0–4
    - col:      0–4
    - pass_loc: 0–4  (0–3 = waiting at loc, 4 = in taxi)
    - dest:     0–3

Fixed locations (R, G, Y, B):
    0 = R = (0, 0)
    1 = G = (0, 4)
    2 = Y = (4, 0)
    3 = B = (4, 3)

Exposed attributes (mirror of MultiPassengerTaxiEnv):
    _taxi_row    int
    _taxi_col    int
    _p_loc       list[int]   length 1 — loc index 0–3, or 4 = in taxi
    _p_dest      list[int]   length 1 — destination loc index 0–3
    _p_delivered list[bool]  length 1
    n_passengers int         always 1
    n_locs       int         always 4
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


# Taxi-v3 fixed locations — must match FIXED_LOCS in multi_passenger_taxi.py
GRID_ROWS = 5
GRID_COLS = 5
FIXED_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
LOC_LABELS = ["R", "G", "Y", "B"]
OBSTACLE_CELLS: frozenset = frozenset()   # Taxi-v3 has walls, not obstacle cells


def _decode_taxi_v3(state: int):
    """Decode a Taxi-v3 state integer into (row, col, pass_loc, dest)."""
    dest     = state % 4;      state //= 4
    pass_loc = state % 5;      state //= 5
    col      = state % 5;      state //= 5
    row      = state
    return row, col, pass_loc, dest


class TaxiV3Wrapper(gym.Wrapper):
    """
    Wraps gym.make('Taxi-v3') and decodes state on every reset/step
    to populate the internal attributes expected by linear_fa.py.
    """

    def __init__(self):
        env = gym.make("Taxi-v3")
        super().__init__(env)

        self.n_passengers = 1
        self.n_locs       = 4

        # Internal state — populated by reset() and step()
        self._taxi_row:    int       = 0
        self._taxi_col:    int       = 0
        self._p_loc:       list[int] = [0]
        self._p_dest:      list[int] = [0]
        self._p_delivered: list[bool] = [False]

    def _sync(self, obs: int) -> None:
        """Decode obs and update internal attributes."""
        row, col, pass_loc, dest = _decode_taxi_v3(obs)
        self._taxi_row    = row
        self._taxi_col    = col
        self._p_loc       = [pass_loc]          # 4 means in-taxi, matches convention
        self._p_dest      = [dest]
        # delivered: passenger was in taxi (pass_loc==4) and is now at dest
        # Taxi-v3 sets pass_loc = dest on successful dropoff and ends episode,
        # so we treat done=True with pass_loc != 4 as delivered.
        # _sync is called before we know done, so we update delivered in step().
        self._p_delivered = [False]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._sync(obs)
        self._p_delivered = [False]
        # Taxi-v3 doesn't always provide action_mask — add one if missing
        if "action_mask" not in info:
            info["action_mask"] = self._compute_action_mask(obs)
        return obs, info

    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)
        self._sync(obs)
        if done:
            self._p_delivered = [True]
        if "action_mask" not in info:
            info["action_mask"] = self._compute_action_mask(obs)
        return obs, reward, done, truncated, info

    def _compute_action_mask(self, obs: int) -> np.ndarray:
        """
        Reconstruct a valid action mask from the Taxi-v3 obs.
        Taxi-v3 returns action_mask in info by default in recent Gymnasium
        versions — this is a fallback in case it doesn't.
        """
        row, col, pass_loc, dest = _decode_taxi_v3(obs)
        mask = np.ones(6, dtype=np.int8)

        # boundaries
        if row == 0:           mask[1] = 0   # can't go North
        if row == GRID_ROWS-1: mask[0] = 0   # can't go South
        if col == 0:           mask[3] = 0   # can't go West
        if col == GRID_COLS-1: mask[2] = 0   # can't go East

        # pickup valid only if on a waiting passenger
        mask[4] = int(pass_loc < 4 and FIXED_LOCS[pass_loc] == (row, col))
        # dropoff valid only if carrying passenger and at destination
        mask[5] = int(pass_loc == 4 and FIXED_LOCS[dest] == (row, col))

        return mask


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = TaxiV3Wrapper()
    obs, info = env.reset(seed=42)

    print("=== TaxiV3Wrapper smoke-test ===")
    print(f"obs:          {obs}")
    print(f"taxi pos:     ({env._taxi_row}, {env._taxi_col})")
    print(f"p_loc:        {env._p_loc}  (4=in taxi)")
    print(f"p_dest:       {env._p_dest}")
    print(f"p_delivered:  {env._p_delivered}")
    print(f"action_mask:  {info['action_mask']}")
    print(f"n_passengers: {env.n_passengers}")
    print(f"n_locs:       {env.n_locs}")

    total_reward = 0.0
    for _ in range(20):
        valid  = np.nonzero(info["action_mask"])[0]
        action = int(np.random.choice(valid))
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"\nReward after up to 20 random steps: {total_reward}")
    print("Smoke-test passed.")