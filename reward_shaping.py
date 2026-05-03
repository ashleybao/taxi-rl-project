"""
reward_shaping.py
=================
Potential-based reward shaping wrapper for MultiPassengerTaxiEnv (5×5)
and LargeMultiPassengerTaxiEnv (15×15).

Usage
-----
    from reward_shaping import ShapedEnv
    from multi_passenger_taxi import MultiPassengerTaxiEnv

    base_env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    env      = ShapedEnv(base_env, gamma=0.95, scale=1.0)

    obs, info = env.reset(seed=42)
    obs, reward, done, truncated, info = env.step(action)
    env.close()

The wrapper is transparent — observation space, action space, action mask,
and encode/decode all pass through to the base env unchanged.

Theory
------
Potential-based shaping (Ng et al., 1999):

    F(s, s') = γ · Φ(s') - Φ(s)

where Φ(s) is a real-valued potential function over states.
Adding F to every transition reward does NOT change the optimal policy
(the shaped and unshaped MDPs have the same optimal π*).

Potential function used here
----------------------------
Φ(s) = -scale · Σ_p  dist(taxi, goal_p)  /  (P · max_dist)

where goal_p is:
    • the passenger's waiting location  (if not yet picked up)
    • the passenger's destination       (if in taxi)
    • 0                                 (if already delivered)

This is always ≤ 0, so γ·Φ(s') - Φ(s) is positive when the agent
moves closer to any unfinished passenger's goal and negative when it
moves away — giving a dense signal every step.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class ShapedEnv(gym.Wrapper):
    """
    Wraps either MultiPassengerTaxiEnv or LargeMultiPassengerTaxiEnv
    and adds potential-based reward shaping.

    Parameters
    ----------
    env   : the base environment instance
    gamma : discount factor (must match the one used in training)
    scale : multiplier on the shaping signal (tune if signal is too weak/strong)
    """

    def __init__(self, env, gamma: float = 0.95, scale: float = 1.0) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale

        # Read grid constants from the wrapped env's module
        import importlib
        m = importlib.import_module(env.__class__.__module__)
        self._fixed_locs = m.FIXED_LOCS
        self._max_dist   = float(m.GRID_ROWS + m.GRID_COLS - 2)
        self._n_locs     = len(self._fixed_locs)

        self._phi_s: float = 0.0   # Φ(s) cached after reset/step

    # ------------------------------------------------------------------
    # Potential function
    # ------------------------------------------------------------------

    def _potential(self) -> float:
        """
        Φ(s) = -scale · mean_over_unfinished_passengers(
                    dist(taxi, passenger_goal) / max_dist
               )

        Returns 0.0 if all passengers are delivered.
        """
        env = self.env
        P   = env.n_passengers
        tr  = env._taxi_row
        tc  = env._taxi_col

        total = 0.0
        n_active = 0

        for p in range(P):
            if env._p_delivered[p]:
                continue
            n_active += 1
            if env._p_loc[p] == self._n_locs:   # in taxi → goal is destination
                goal_idx = env._p_dest[p]
            else:                                 # waiting  → goal is pickup loc
                goal_idx = env._p_loc[p]

            gr, gc = self._fixed_locs[goal_idx]
            total += (abs(tr - gr) + abs(tc - gc)) / self._max_dist

        if n_active == 0:
            return 0.0

        return -self.scale * (total / n_active)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._phi_s = self._potential()
        return obs, info

    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)

        phi_s_prime  = self._potential()
        shaping      = self.gamma * phi_s_prime - self._phi_s
        self._phi_s  = phi_s_prime

        shaped_reward = reward + shaping
        return obs, shaped_reward, done, truncated, info

    # ------------------------------------------------------------------
    # Pass-throughs so linear_fa.py can still read internal state
    # ------------------------------------------------------------------

    @property
    def _taxi_row(self):    return self.env._taxi_row
    @property
    def _taxi_col(self):    return self.env._taxi_col
    @property
    def _p_loc(self):       return self.env._p_loc
    @property
    def _p_dest(self):      return self.env._p_dest
    @property
    def _p_delivered(self): return self.env._p_delivered
    @property
    def n_passengers(self): return self.env.n_passengers

    def _encode(self):          return self.env._encode()
    def _decode(self, state):   return self.env._decode(state)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multi_passenger_taxi import MultiPassengerTaxiEnv

    base = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    env  = ShapedEnv(base, gamma=0.95, scale=1.0)

    obs, info = env.reset(seed=42)
    print(f"Initial Φ(s) = {env._phi_s:.4f}")

    total_raw    = 0.0
    total_shaped = 0.0

    for step in range(10):
        valid  = np.nonzero(info["action_mask"])[0]
        action = int(np.random.choice(valid))

        obs, shaped_reward, done, truncated, info = env.step(action)

        # also get raw reward for comparison
        # (we can infer it: raw = shaped - shaping = shaped - (γΦ' - Φ))
        # easier to just print shaped and note shaping = shaped - (-1 or -10 or +20)
        total_shaped += shaped_reward
        print(f"  step {step+1:2d} | action={action} "
              f"shaped_r={shaped_reward:+.3f}  Φ(s')={env._phi_s:.4f}")

        if done or truncated:
            break

    env.close()
    print("\nSmoke-test passed.")