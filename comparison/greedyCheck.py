# import numpy as np

# # q_ql    = np.load("/home/common/ji-bao-lin/taxi/results/q_learning/checkpoints/best_multi_masked_run.npz")["q_table"] <- prev Jiyuan's
# q_ql = np.load("/home/common/ji-bao-lin/taxi/best_multi_masked_run.npz")["q_table"] # new multi
# q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa/multi/best_multi_masked_run.npz")["q_table"]


# # Reachable = states either algorithm actually updated
# reachable = (q_ql.sum(axis=1) != 0) | (q_sarsa.sum(axis=1) != 0)

# policy_ql    = np.argmax(q_ql,    axis=1)
# policy_sarsa = np.argmax(q_sarsa, axis=1)

# agreement = (policy_ql[reachable] == policy_sarsa[reachable]).mean()
# print(f"Reachable states: {reachable.sum()} / {len(reachable)}")
# print(f"Policy agreement: {agreement:.1%}")

import numpy as np

q_ql    = np.load("/home/common/ji-bao-lin/taxi/results/q_learning_decay/multi/best_multi_masked_run.npz")["q_table"]
q_sarsa = np.load("/home/common/ji-bao-lin/taxi/results/sarsa_decay/multi/best_multi_masked_run.npz")["q_table"]

# Need the env to get action masks per state — but the action mask depends on state, not on the Q-table.
# We can recover it by running episodes and recording (state, mask) pairs.
import sys
sys.path.insert(0, "/home/common/ji-bao-lin/taxi")
from multi_passenger_taxi import MultiPassengerTaxiEnv

env = MultiPassengerTaxiEnv(n_passengers=2, max_steps=200)
state_to_mask = {}
for ep in range(2000):
    state, info = env.reset(seed=58922320 + ep)
    state_to_mask[state] = info["action_mask"]
    done = trunc = False
    while not (done or trunc):
        valid = np.nonzero(info["action_mask"] == 1)[0]
        action = int(valid[np.argmax(q_ql[state, valid])])  # roll out QL greedy to collect states
        state, _, done, trunc, info = env.step(action)
        if not (done or trunc):
            state_to_mask[state] = info["action_mask"]
env.close()

# Now compute masked argmax for each visited state, both algorithms
agree = 0
total = 0
qval_diffs = []
for s, mask in state_to_mask.items():
    valid = np.nonzero(mask == 1)[0]
    if len(valid) == 0:
        continue
    a_ql    = int(valid[np.argmax(q_ql[s, valid])])
    a_sarsa = int(valid[np.argmax(q_sarsa[s, valid])])
    total += 1
    if a_ql == a_sarsa:
        agree += 1
    qval_diffs.append(abs(q_ql[s, a_ql] - q_sarsa[s, a_sarsa]))

print(f"Reachable states visited: {total}")
print(f"Masked-argmax policy agreement: {agree/total:.1%}")
print(f"Mean |Q_QL - Q_SARSA| at chosen action: {np.mean(qval_diffs):.3f}")