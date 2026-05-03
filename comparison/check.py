import numpy as np
import matplotlib.pyplot as plt

ql_rewards    = np.load("/home/common/ji-bao-lin/taxi/best_multi_masked_run.npz")["episode_rewards"]
sarsa_rewards = np.load("/home/common/ji-bao-lin/taxi/results/sarsa/multi/best_multi_masked_run.npz")["episode_rewards"]

# Smooth with a rolling mean
def smooth(x, w=500):
    return np.convolve(x, np.ones(w)/w, mode="valid")

plt.plot(smooth(ql_rewards),    label="Q-learning")
plt.plot(smooth(sarsa_rewards), label="SARSA")
plt.xlabel("Episode"); plt.ylabel("Reward (rolling mean)"); plt.legend()
plt.savefig("learning_curves.png", dpi=120)
print("saved learning_curves.png")