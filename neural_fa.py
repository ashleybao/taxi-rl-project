"""
neural_fa.py
============
One-hidden-layer neural network Q-learning.

Phase 1 : train on Gymnasium Taxi-v3 (5×5) via TaxiV3Wrapper.
Phase 2 : zero-shot eval on 15×15 with Phase 1 weights.
Phase 3 : fine-tune (warm ε) vs scratch (cold ε) on 15×15.

Because D=13 is identical for both envs (transferable features only),
the network architecture is the same size and weights CAN be copied
directly — unlike the old multi-passenger script where D differed.

Architecture
------------
    φ(s) ∈ ℝ^13
        → Linear(13, hidden_dim) → ReLU
        → Linear(hidden_dim, 6)
        → Q(s, ·) ∈ ℝ^6

Training
--------
Semi-gradient TD(0) with optional experience replay.

    target  = r + γ · max_{a'∈valid} Q(s'; θ)   (no grad)
    loss    = (Q(s; θ)[a] - target)²
    θ ← θ - α · ∇_θ loss

Outputs (saved to results/neural_fa/)
--------------------------------------
    training_taxiv3.png             reward curve + ε, Phase 1
    weights_taxiv3.png              first-layer heatmap, Phase 1
    training_15x15_finetune.png     reward curve, Phase 3 fine-tune
    training_15x15_scratch.png      reward curve, Phase 3 scratch
    weights_15x15_finetuned.png     first-layer heatmap, best fine-tune
    transfer_comparison.png         fine-tune vs scratch on 15×15
    models/taxiv3_run{n}.pt         per-run state_dicts, Phase 1
    models/15x15_finetune_run{n}.pt per-run state_dicts, Phase 3 fine-tune
    models/15x15_scratch_run{n}.pt  per-run state_dicts, Phase 3 scratch
    best_taxiv3.pt                  best Phase 1 model state_dict
    best_15x15_finetune.pt          best fine-tune model state_dict
    best_15x15_scratch.pt           best scratch model state_dict
    summary.json                    per-run stats + zero-shot result
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from linear_fa import (
    EnvConfig,
    feature_dim,
    features_from_env,
    linear_decay,
    exponential_decay,
    build_feature_names,
)

N_ACTIONS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(phi).float().unsqueeze(0)
            return self.forward(t).squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.buf: list = []
        self.pos = 0

    def push(self, phi, action, reward, phi_next, done) -> None:
        item = (phi, action, reward, phi_next, done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        phis, actions, rewards, phis_next, dones = zip(*batch)
        return (
            np.array(phis,      dtype=np.float32),
            np.array(actions,   dtype=np.int64),
            np.array(rewards,   dtype=np.float32),
            np.array(phis_next, dtype=np.float32),
            np.array(dones,     dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_neural_fa(
    env,
    cfg:               EnvConfig,
    episodes:          int             = 10_000,
    hidden_dim:        int             = 64,
    lr:                float           = 1e-3,
    gamma:             float           = 0.95,
    epsilon:           float           = 0.1,
    epsilon_schedule:  Callable | None = None,
    use_action_mask:   bool            = True,
    seed:              int             = 42,
    track_epsilon:     bool            = False,
    log_interval:      int             = 500,
    use_replay:        bool            = False,
    replay_capacity:   int             = 10_000,
    batch_size:        int             = 64,
    replay_start:      int             = 500,
    model_init:        QNetwork | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    D = feature_dim(cfg)
    model = QNetwork(D, hidden_dim)
    if model_init is not None:
        model.load_state_dict(model_init.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    buffer    = ReplayBuffer(replay_capacity) if use_replay else None

    episode_rewards: list[float] = []
    epsilon_trace:   list[float] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        phi       = features_from_env(env, cfg)

        total_reward = 0.0
        done = truncated = False

        eps = epsilon_schedule(ep) if epsilon_schedule else epsilon
        if track_epsilon:
            epsilon_trace.append(eps)

        while not (done or truncated):
            mask  = info["action_mask"] if use_action_mask \
                    else np.ones(N_ACTIONS, dtype=np.int8)
            valid = np.nonzero(mask)[0]

            if np.random.random() < eps:
                action = int(np.random.choice(valid))
            else:
                q      = model.q_values(phi)
                action = int(valid[np.argmax(q[valid])])

            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            phi_next = features_from_env(env, cfg)

            if use_replay:
                buffer.push(phi, action, reward, phi_next,
                            float(done or truncated))
                phi = phi_next
                if len(buffer) >= replay_start:
                    _update_from_replay(model, optimizer, loss_fn, buffer,
                                        batch_size, gamma, use_action_mask,
                                        info)
            else:
                _online_update(model, optimizer, loss_fn,
                               phi, action, reward, phi_next,
                               done or truncated, gamma, info, use_action_mask)
                phi = phi_next

        episode_rewards.append(total_reward)

        if log_interval and (ep + 1) % log_interval == 0:
            recent = np.mean(episode_rewards[-log_interval:])
            print(f"  ep {ep+1:>6} | mean reward (last {log_interval}): "
                  f"{recent:+.1f}  ε={eps:.3f}")

    return {
        "model":           model,
        "episode_rewards": episode_rewards,
        "mean_reward":     float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "epsilons":        epsilon_trace if track_epsilon else None,
        "feature_dim":     D,
        "hidden_dim":      hidden_dim,
    }


def _online_update(model, optimizer, loss_fn,
                   phi, action, reward, phi_next,
                   terminal, gamma, info, use_action_mask):
    phi_t      = torch.from_numpy(phi).float().unsqueeze(0)
    phi_next_t = torch.from_numpy(phi_next).float().unsqueeze(0)

    with torch.no_grad():
        q_next = model(phi_next_t).squeeze(0).numpy()
        if terminal:
            target = reward
        else:
            next_mask  = info["action_mask"] if use_action_mask \
                         else np.ones(N_ACTIONS, dtype=np.int8)
            next_valid = np.nonzero(next_mask)[0]
            target = reward + gamma * float(np.max(q_next[next_valid])) \
                     if len(next_valid) > 0 else reward

    q_pred = model(phi_t).squeeze(0)[action]
    loss   = loss_fn(q_pred, torch.tensor(target, dtype=torch.float32))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def _update_from_replay(model, optimizer, loss_fn, buffer,
                         batch_size, gamma, use_action_mask, info):
    phis, actions, rewards, phis_next, dones = buffer.sample(batch_size)

    phi_t      = torch.from_numpy(phis).float()
    phi_next_t = torch.from_numpy(phis_next).float()
    rewards_t  = torch.from_numpy(rewards).float()
    dones_t    = torch.from_numpy(dones).float()

    with torch.no_grad():
        q_next    = model(phi_next_t)
        best_next = q_next.max(dim=1).values
        targets   = rewards_t + gamma * best_next * (1 - dones_t)

    q_pred  = model(phi_t)
    q_taken = q_pred.gather(
        1, torch.from_numpy(actions).long().unsqueeze(1)
    ).squeeze(1)

    loss = loss_fn(q_taken, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    env,
    cfg:             EnvConfig,
    model:           QNetwork,
    n_episodes:      int  = 500,
    seed:            int  = 9999,
    use_action_mask: bool = True,
) -> dict:
    model.eval()
    rewards, lengths = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        phi = features_from_env(env, cfg)
        total_reward = 0.0
        steps = 0
        done = truncated = False

        while not (done or truncated):
            mask  = info["action_mask"] if use_action_mask \
                    else np.ones(N_ACTIONS, dtype=np.int8)
            valid = np.nonzero(mask)[0]
            q     = model.q_values(phi)
            action = int(valid[np.argmax(q[valid])])

            obs, reward, done, truncated, info = env.step(action)
            phi = features_from_env(env, cfg)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    model.train()
    env.close()
    return {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_length":  float(np.mean(lengths)),
        "success_rate": float(np.mean([r > 0 for r in rewards])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(all_results: list[dict], cfg: EnvConfig,
                  output_dir: Path, tag: str = "") -> None:
    def smooth(x, w=200):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    curves     = np.array([r["episode_rewards"] for r in all_results])
    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sm  = smooth(mean_curve)
    slo = smooth(mean_curve - std_curve)
    shi = smooth(mean_curve + std_curve)
    x   = np.arange(len(sm))

    ax1.plot(x, sm, color="#1D9E75", label="Mean reward (smoothed)")
    ax1.fill_between(x, slo, shi, alpha=0.2, color="#1D9E75", label="±1 std")
    ax1.set_ylabel("Episode reward")
    ax1.set_title(
        f"Neural FA Q-learning — {cfg.n_passengers}p  "
        f"{cfg.grid_rows}×{cfg.grid_cols}  "
        f"D={feature_dim(cfg)}  hidden={all_results[0]['hidden_dim']}"
        + (f"  [{tag}]" if tag else "")
    )
    ax1.legend(); ax1.grid(True, alpha=0.3)

    eps_traces = [r["epsilons"] for r in all_results if r["epsilons"]]
    if eps_traces:
        ax2.plot(np.mean(eps_traces, axis=0), color="darkorange", label="ε")
        ax2.set_ylabel("Epsilon ε")
        ax2.set_xlabel("Episode")
        ax2.set_ylim(0, 1.05)
        ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_dir / (f"training_{tag}.png" if tag else "training.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_first_layer_weights(model: QNetwork, cfg: EnvConfig,
                              output_dir: Path, tag: str = "") -> None:
    W     = model.net[0].weight.detach().numpy()   # (hidden, D)
    names = build_feature_names(cfg)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    im = ax.imshow(W, aspect="auto", cmap="RdYlGn",
                   vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("Hidden unit")
    ax.set_title(f"First layer weights  [{tag}]" if tag
                 else "First layer weights")
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    fname = output_dir / (f"weights_{tag}.png" if tag else "weights.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main: Taxi-v3 (5×5) → SinglePassengerLargeTaxiEnv (15×15) transfer
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from taxi_v3_wrapper import TaxiV3Wrapper
    from single_passenger_taxi_large import SinglePassengerLargeTaxiEnv

    output_dir  = Path("results/neural_fa")
    models_dir  = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    HIDDEN_DIM = 64
    N_RUNS     = 5
    BASE_SEED  = 58922320
    seeds      = [BASE_SEED + i for i in range(N_RUNS)]

    # ── PHASE 1: train on Taxi-v3 (5×5) ─────────────────────────────────────
    EPISODES_SMALL = 10_000
    schedule_small = linear_decay(EPISODES_SMALL, eps_start=1.0,
                                  eps_end=0.05, decay_frac=0.95)

    _probe    = TaxiV3Wrapper()
    cfg_small = EnvConfig.from_env(_probe)
    _probe.close()

    print(f"Feature dim D={feature_dim(cfg_small)}  hidden={HIDDEN_DIM}  "
          f"(same for both envs — weight transfer is direct)\n")
    print("=== PHASE 1: Neural FA on Taxi-v3 (5×5) ===")

    small_results    = []
    small_eval_stats = []
    best_small_model  = None
    best_small_reward = -np.inf
    best_small_idx    = 0

    for run in range(N_RUNS):
        seed = seeds[run]
        print(f"─── Run {run+1}/{N_RUNS}  (seed {seed}) ───")

        env    = TaxiV3Wrapper()
        result = train_neural_fa(
            env, cfg_small,
            episodes         = EPISODES_SMALL,
            hidden_dim       = HIDDEN_DIM,
            lr               = 5e-4,
            gamma            = 0.95,
            epsilon_schedule = schedule_small,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = True,
            log_interval     = 1000,
            use_replay       = True,
            replay_capacity  = 20_000,
            batch_size       = 64,
            replay_start     = 500,
        )
        eval_env   = TaxiV3Wrapper()
        eval_stats = evaluate_policy(eval_env, cfg_small, result["model"],
                                     n_episodes=500, seed=seed + 10000)
        print(f"  EVAL → mean: {eval_stats['mean_reward']:+.1f} ± "
              f"{eval_stats['std_reward']:.1f}  "
              f"success: {eval_stats['success_rate']:.1%}  "
              f"len: {eval_stats['mean_length']:.0f}\n")

        small_results.append(result)
        small_eval_stats.append(eval_stats)

        # per-run model
        torch.save(result["model"].state_dict(),
                   models_dir / f"taxiv3_run{run+1}.pt")

        if eval_stats["mean_reward"] > best_small_reward:
            best_small_reward = eval_stats["mean_reward"]
            best_small_model  = result["model"]
            best_small_idx    = run

    plot_training(small_results, cfg_small, output_dir, tag="taxiv3")
    plot_first_layer_weights(best_small_model, cfg_small, output_dir, tag="taxiv3")
    torch.save(best_small_model.state_dict(), output_dir / "best_taxiv3.pt")

    # summary — Phase 1
    summary = {
        "taxiv3": [
            {
                "run":          run + 1,
                "seed":         seeds[run],
                "mean_reward":  small_results[run]["mean_reward"],
                "std_reward":   small_results[run]["std_reward"],
                "eval_mean":    small_eval_stats[run]["mean_reward"],
                "eval_std":     small_eval_stats[run]["std_reward"],
                "eval_success": small_eval_stats[run]["success_rate"],
            }
            for run in range(N_RUNS)
        ]
    }

    # ── PHASE 2: zero-shot eval on 15×15 ─────────────────────────────────────
    _probe    = SinglePassengerLargeTaxiEnv()
    cfg_large = EnvConfig.from_env(_probe)
    _probe.close()

    D_small = feature_dim(cfg_small)
    D_large = feature_dim(cfg_large)

    print(f"\n=== PHASE 2: zero-shot transfer check ===")
    print(f"Taxi-v3 D={D_small}  |  15×15 D={D_large}")

    assert D_small == D_large, (
        f"Dim mismatch ({D_small} vs {D_large}) — "
        "both envs must have the same n_passengers."
    )

    print("Dims match — copying Phase 1 weights directly into 15×15 network.")
    env_zs    = SinglePassengerLargeTaxiEnv(max_steps=500)
    zs_stats  = evaluate_policy(env_zs, cfg_large, best_small_model,
                                n_episodes=500, seed=BASE_SEED + 99999)
    print(f"Zero-shot → mean: {zs_stats['mean_reward']:+.1f} ± "
          f"{zs_stats['std_reward']:.1f}  "
          f"success: {zs_stats['success_rate']:.1%}\n")

    summary["zero_shot"] = {
        "mean_reward":  zs_stats["mean_reward"],
        "std_reward":   zs_stats["std_reward"],
        "success_rate": zs_stats["success_rate"],
        "mean_length":  zs_stats["mean_length"],
    }

    # ── PHASE 3: fine-tune vs scratch on 15×15 ───────────────────────────────
    EPISODES_LARGE   = 20_000
    schedule_ft      = linear_decay(EPISODES_LARGE, eps_start=0.3,
                                    eps_end=0.05, decay_frac=0.7)
    schedule_scratch = linear_decay(EPISODES_LARGE, eps_start=1.0,
                                    eps_end=0.05, decay_frac=0.8)

    print(f"=== PHASE 3: fine-tune vs scratch on 15×15 ({EPISODES_LARGE} eps) ===\n")

    finetune_results = []
    scratch_results  = []
    finetune_eval    = []
    scratch_eval     = []
    best_ft_model    = None
    best_ft_reward   = -np.inf
    best_sc_model    = None
    best_sc_reward   = -np.inf

    for run in range(N_RUNS):
        seed = seeds[run]
        print(f"─── Run {run+1}/{N_RUNS}  (seed {seed}) ───")

        # Fine-tune: copy weights from Phase 1 best model, warm ε
        env_ft    = SinglePassengerLargeTaxiEnv(max_steps=500)
        result_ft = train_neural_fa(
            env_ft, cfg_large,
            episodes         = EPISODES_LARGE,
            hidden_dim       = HIDDEN_DIM,
            lr               = 3e-4,
            gamma            = 0.95,
            epsilon_schedule = schedule_ft,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = True,
            log_interval     = 2000,
            use_replay       = True,
            replay_capacity  = 30_000,
            batch_size       = 128,
            replay_start     = 1000,
            model_init       = best_small_model,   # transfer weights
        )
        eval_ft_env = SinglePassengerLargeTaxiEnv(max_steps=500)
        eval_ft     = evaluate_policy(eval_ft_env, cfg_large, result_ft["model"],
                                      n_episodes=500, seed=seed + 20000)
        print(f"  Fine-tune EVAL → mean: {eval_ft['mean_reward']:+.1f}  "
              f"success: {eval_ft['success_rate']:.1%}")
        finetune_results.append(result_ft)
        finetune_eval.append(eval_ft)

        torch.save(result_ft["model"].state_dict(),
                   models_dir / f"15x15_finetune_run{run+1}.pt")

        if eval_ft["mean_reward"] > best_ft_reward:
            best_ft_reward = eval_ft["mean_reward"]
            best_ft_model  = result_ft["model"]

        # Scratch: random init, cold ε
        env_sc    = SinglePassengerLargeTaxiEnv(max_steps=500)
        result_sc = train_neural_fa(
            env_sc, cfg_large,
            episodes         = EPISODES_LARGE,
            hidden_dim       = HIDDEN_DIM,
            lr               = 3e-4,
            gamma            = 0.95,
            epsilon_schedule = schedule_scratch,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = False,
            log_interval     = 2000,
            use_replay       = True,
            replay_capacity  = 30_000,
            batch_size       = 128,
            replay_start     = 1000,
            model_init       = None,
        )
        eval_sc_env = SinglePassengerLargeTaxiEnv(max_steps=500)
        eval_sc     = evaluate_policy(eval_sc_env, cfg_large, result_sc["model"],
                                      n_episodes=500, seed=seed + 20000)
        print(f"  Scratch   EVAL → mean: {eval_sc['mean_reward']:+.1f}  "
              f"success: {eval_sc['success_rate']:.1%}\n")
        scratch_results.append(result_sc)
        scratch_eval.append(eval_sc)

        torch.save(result_sc["model"].state_dict(),
                   models_dir / f"15x15_scratch_run{run+1}.pt")

        if eval_sc["mean_reward"] > best_sc_reward:
            best_sc_reward = eval_sc["mean_reward"]
            best_sc_model  = result_sc["model"]

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_training(finetune_results, cfg_large, output_dir, tag="15x15_finetune")
    plot_training(scratch_results,  cfg_large, output_dir, tag="15x15_scratch")
    plot_first_layer_weights(best_ft_model, cfg_large, output_dir,
                             tag="15x15_finetuned")

    def smooth(x, w=300):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ft_mean = np.mean([r["episode_rewards"] for r in finetune_results], axis=0)
    sc_mean = np.mean([r["episode_rewards"] for r in scratch_results],  axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(smooth(ft_mean), label="Fine-tune (Taxi-v3 → 15×15)", color="#1D9E75")
    plt.plot(smooth(sc_mean), label="Scratch (15×15)",              color="#D85A30")
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.title("Neural FA transfer: Taxi-v3 → 15×15  vs  from scratch")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── best-run checkpoints ──────────────────────────────────────────────────
    torch.save(best_ft_model.state_dict(), output_dir / "best_15x15_finetune.pt")
    torch.save(best_sc_model.state_dict(), output_dir / "best_15x15_scratch.pt")

    # ── summary.json ──────────────────────────────────────────────────────────
    summary["15x15_finetune"] = [
        {
            "run":          run + 1,
            "seed":         seeds[run],
            "mean_reward":  finetune_results[run]["mean_reward"],
            "std_reward":   finetune_results[run]["std_reward"],
            "eval_mean":    finetune_eval[run]["mean_reward"],
            "eval_std":     finetune_eval[run]["std_reward"],
            "eval_success": finetune_eval[run]["success_rate"],
        }
        for run in range(N_RUNS)
    ]
    summary["15x15_scratch"] = [
        {
            "run":          run + 1,
            "seed":         seeds[run],
            "mean_reward":  scratch_results[run]["mean_reward"],
            "std_reward":   scratch_results[run]["std_reward"],
            "eval_mean":    scratch_eval[run]["mean_reward"],
            "eval_std":     scratch_eval[run]["std_reward"],
            "eval_success": scratch_eval[run]["success_rate"],
        }
        for run in range(N_RUNS)
    ]

    json.dump(summary, open(output_dir / "summary.json", "w"), indent=2)

    print("\nDone. Files written to", output_dir)
    print(f"  Plots      : training_taxiv3.png, weights_taxiv3.png")
    print(f"               training_15x15_finetune.png, training_15x15_scratch.png")
    print(f"               weights_15x15_finetuned.png, transfer_comparison.png")
    print(f"  Per-run    : models/taxiv3_run{{1-{N_RUNS}}}.pt")
    print(f"               models/15x15_finetune_run{{1-{N_RUNS}}}.pt")
    print(f"               models/15x15_scratch_run{{1-{N_RUNS}}}.pt")
    print(f"  Best runs  : best_taxiv3.pt, best_15x15_finetune.pt, best_15x15_scratch.pt")
    print(f"  Summary    : summary.json")