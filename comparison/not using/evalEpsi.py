## this is used to verify if the q-table is correct
## epsion 0.0 should always beat epsion 0.1, but if the q-table is wrong, it might not
#           no-decay            decay
# ε=0.1     +11.75 / 99.8%      +12.71 / 100%
# ε=0.0     +15.24 / 100%       +15.71 / 100%
# so its verified!

import sys
import numpy as np

sys.path.insert(0, "/home/common/ji-bao-lin/taxi")
from multi_passenger_taxi import MultiPassengerTaxiEnv

# q_ql    = np.load("/home/common/ji-bao-lin/taxi/best_multi_masked_run.npz")["q_table"]
# q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa/multi/best_multi_masked_run.npz")["q_table"]

# decay
q_ql    = np.load("/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/best_multi_masked_run.npz")["q_table"]
q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/best_multi_masked_run.npz")["q_table"]

EPSION = 0.0

def evaluate_eps(q_table, n_episodes=2000, seed=58922320, epsilon=EPSION):
    env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    returns, successes = [], []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep)
        total, done, trunc = 0.0, False, False
        while not (done or trunc):
            mask = info["action_mask"]
            valid = np.nonzero(mask == 1)[0]
            if rng.random() < epsilon:
                action = int(rng.choice(valid))
            else:
                action = int(valid[np.argmax(q_table[state, valid])])
            state, r, done, trunc, info = env.step(action)
            total += r
        returns.append(total)
        successes.append(done and not trunc)
    env.close()
    return np.mean(returns), np.mean(successes)


for name, q in [("Q-learning", q_ql), ("SARSA", q_sarsa)]:
    r, s = evaluate_eps(q, epsilon=EPSION)
    print(f"{name:10s} eps={EPSION}  reward: {r:7.2f}  success: {s:.1%}")