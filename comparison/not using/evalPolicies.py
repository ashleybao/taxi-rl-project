import numpy as np
import sys
sys.path.insert(0, "/home/common/ji-bao-lin/taxi")
from multi_passenger_taxi import MultiPassengerTaxiEnv

# ----------------- CONFIG -----------------
USE_MASK = False   # True = restrict argmax to valid actions; False = argmax over all actions
USE_DECAY_QTABLES = False
# ------------------------------------------

if USE_DECAY_QTABLES:
    q_ql    = np.load("/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/best_multi_masked_run.npz")["q_table"]
    q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/best_multi_masked_run.npz")["q_table"]
else:
    q_ql    = np.load("/home/common/ji-bao-lin/taxi/best_multi_masked_run.npz")["q_table"]
    q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa/multi/best_multi_masked_run.npz")["q_table"]


def evaluate(q_table, n_episodes=2000, seed=58922320, use_mask=USE_MASK):
    env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
    returns, lengths, successes = [], [], []
    for ep in range(n_episodes):
        state, info = env.reset(seed=seed + ep)
        total, steps, done, trunc = 0.0, 0, False, False
        while not (done or trunc):
            if use_mask:
                mask = info["action_mask"]
                valid = np.nonzero(mask == 1)[0]
                action = int(valid[np.argmax(q_table[state, valid])])
            else:
                action = int(np.argmax(q_table[state]))
            state, r, done, trunc, info = env.step(action)
            total += r
            steps += 1
        returns.append(total)
        lengths.append(steps)
        successes.append(done and not trunc)
    env.close()
    return np.mean(returns), np.std(returns), np.mean(lengths), np.mean(successes)


print(f"USE_MASK = {USE_MASK}, USE_DECAY_QTABLES = {USE_DECAY_QTABLES}")
for name, q in [("Q-learning", q_ql), ("SARSA", q_sarsa)]:
    mean_r, std_r, mean_len, succ = evaluate(q)
    print(f"{name:10s}  reward: {mean_r:7.2f} ± {std_r:.2f}   "
          f"length: {mean_len:5.1f}   success: {succ:.1%}")