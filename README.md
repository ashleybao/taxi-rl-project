# 🚕 Multi-Passenger Taxi Reinforcement Learning

This repository implements reinforcement learning agents (Q-learning and SARSA) for a **multi-passenger extension of the Taxi-v3 environment**, including action masking, exploration strategies, and visualization tools.

---

## 📌 Overview

We study how different reinforcement learning configurations affect performance in a constrained navigation environment.

### Algorithms
- **Q-learning (off-policy)**
- **SARSA (on-policy)**

### Environment
- Custom `MultiPassengerTaxiEnv`
- Extension of the classic Taxi-v3 problem
- Multiple passengers with fixed pickup/dropoff locations
- Action masking for invalid moves

### Key Features
- Action masking (valid action filtering)
- Static vs decaying ε-greedy exploration
- Multi-seed evaluation (12 runs per configuration)
- Q-table saving and analysis
- GIF-based policy visualization

---

## 🧠 Environment Description

The environment is a 5×5 grid world:

- Taxi must pick up and drop off multiple passengers
- Each passenger has a fixed pickup/dropoff location
- Invalid actions can be masked using `action_mask`
- Episode ends when all passengers are delivered or max steps reached

---

## ⚙️ Algorithms

### Q-Learning (off-policy)

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]
\]

---

### SARSA (on-policy)

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma Q(s',a') - Q(s,a)\big]
\]

**Key difference:**
- Q-learning uses the greedy next action (max)
- SARSA uses the *actual next action sampled from the policy*

---

## 🧪 Experimental Setup

Each experiment runs:

- 12 independent random seeds
- 50,000 episodes per run
- Fixed hyperparameters:
  - Learning rate: `0.1`
  - Discount factor: `0.95`
  - Epsilon: `0.1` (static or scheduled depending on experiment)
  - Max steps per episode: `200`

---


---

## 📊 Outputs

### 1. Training Logs
- `multi_summary.json` → per-run statistics (mean/std reward)
- `multi_summary.txt` → human-readable summary

### 2. Q-tables
- `multi_*_qtables.npy`
  - Shape: `(n_runs, n_states, n_actions)`

### 3. Best Runs
- `best_multi_*_run.npz`
  - `q_table`
  - `episode_rewards`
  - `mean_reward`
  - `std_reward`
  - `seed`
  - `run_index`

---

## 🎞️ Visualization

Each trained policy can be rendered as a GIF:

- Greedy policy execution (`ε = 0`)
- Action-masked decision making
- Multi-passenger pickup/dropoff behavior
