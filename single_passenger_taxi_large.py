"""
single_passenger_taxi_large.py
===============================
A convenience wrapper around LargeMultiPassengerTaxiEnv fixed to 1 passenger.

15×15 grid, 8 pickup/dropoff stations, obstacle city-blocks.
Identical reward structure, action mask, and state encoding to the
multi-passenger version — just n_passengers=1 baked in.

Stations:
    0=NW (0,0)   1=NE (0,14)
    2=MW (4,2)   3=MC (4,7)   4=ME (4,12)
    5=SW (10,2)  6=SC (10,7)  7=SE (14,14)
"""

from multi_passenger_taxi_large import (
    LargeMultiPassengerTaxiEnv,
    GRID_ROWS,
    GRID_COLS,
    FIXED_LOCS,
    LOC_LABELS,
    OBSTACLE_CELLS,
)


class SinglePassengerLargeTaxiEnv(LargeMultiPassengerTaxiEnv):
    """15×15 taxi environment with exactly one passenger."""

    def __init__(self, max_steps: int = 500) -> None:
        super().__init__(n_passengers=1, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    env = SinglePassengerLargeTaxiEnv(max_steps=500)
    obs, info = env.reset(seed=0)

    print("=== SinglePassengerLargeTaxiEnv ===")
    print(f"Grid          : {GRID_ROWS}×{GRID_COLS}")
    print(f"Obstacle cells: {len(OBSTACLE_CELLS)}")
    print(f"Stations      : {len(FIXED_LOCS)}")
    print(f"Obs space     : {env.observation_space.n:,}")
    print(f"Feature dim   : {env.feature_dim}")
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