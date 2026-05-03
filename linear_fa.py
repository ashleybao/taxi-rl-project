"""
linear_fa.py
============
Linear function-approximation Q-learning.
Trains on Gymnasium Taxi-v3 (5×5) then transfers to SinglePassengerLargeTaxiEnv (15×15).

Q(s, a) ≈ w_a · φ(s)

W ∈ ℝ^{6 × D}.  Semi-gradient TD(0) update per step:
    δ  = r + γ · max_{a'∈valid} w_{a'} · φ(s') - w_a · φ(s)
    w_a ← w_a + α · δ · φ(s)

Transferable feature groups only (no station-dependent features):
─────────────────────────────────────────────────────────────────
GROUP 0 — taxi position (2)
GROUP 1 — signed direction to next goal (2P)
GROUP 2 — normalised Manhattan distance to goal (P)
GROUP 3 — obstacle proximity, cardinal dirs, radius 2 (4)
GROUP 4 — progress fractions (3)
GROUP 5 — bias (1)

Total D = 10 + 3P
For P=1: D = 13  — identical for Taxi-v3 and the 15×15 env.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


N_ACTIONS  = 6
OBS_RADIUS = 2


# ─────────────────────────────────────────────────────────────────────────────
# EnvConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnvConfig:
    grid_rows:      int
    grid_cols:      int
    fixed_locs:     tuple[tuple[int, int], ...]
    obstacle_cells: frozenset[tuple[int, int]]
    n_passengers:   int
    loc_labels:     tuple[str, ...]

    @property
    def n_locs(self) -> int:
        return len(self.fixed_locs)

    @property
    def max_dist(self) -> float:
        return float(self.grid_rows + self.grid_cols - 2)

    @classmethod
    def from_env(cls, env) -> "EnvConfig":
        import importlib
        m = importlib.import_module(env.__class__.__module__)

        fixed_locs     = tuple(tuple(loc) for loc in m.FIXED_LOCS)
        obstacle_cells = frozenset(getattr(m, "OBSTACLE_CELLS", frozenset()))
        loc_labels     = tuple(
            getattr(m, "LOC_LABELS", [str(i) for i in range(len(fixed_locs))])
        )
        return cls(
            grid_rows      = m.GRID_ROWS,
            grid_cols      = m.GRID_COLS,
            fixed_locs     = fixed_locs,
            obstacle_cells = obstacle_cells,
            n_passengers   = env.n_passengers,
            loc_labels     = loc_labels,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature dimension
# ─────────────────────────────────────────────────────────────────────────────

def feature_dim(cfg: EnvConfig) -> int:
    """D = 10 + 3P — independent of grid size and number of stations."""
    P = cfg.n_passengers
    return 2 + 2 * P + P + 4 + 3 + 1


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    taxi_row:    int,
    taxi_col:    int,
    p_locs:      list[int],
    p_dests:     list[int],
    p_delivered: list[bool],
    cfg:         EnvConfig,
) -> np.ndarray:
    P  = cfg.n_passengers
    L  = cfg.n_locs
    R  = cfg.grid_rows
    C  = cfg.grid_cols
    FL = cfg.fixed_locs
    OC = cfg.obstacle_cells
    MD = cfg.max_dist

    parts: list[np.ndarray] = []

    # GROUP 0: taxi position (2)
    parts.append(np.array([taxi_row / (R - 1), taxi_col / (C - 1)],
                           dtype=np.float32))

    # GROUP 1: signed direction to next goal (2P)
    for p in range(P):
        if p_delivered[p]:
            parts.append(np.zeros(2, dtype=np.float32))
        else:
            target_loc = p_dests[p] if p_locs[p] == L else p_locs[p]
            gr, gc = FL[target_loc]
            parts.append(np.array([
                (gr - taxi_row) / (R - 1),
                (gc - taxi_col) / (C - 1),
            ], dtype=np.float32))

    # GROUP 2: normalised Manhattan distance to goal (P)
    dists = np.zeros(P, dtype=np.float32)
    for p in range(P):
        if not p_delivered[p]:
            target_loc = p_dests[p] if p_locs[p] == L else p_locs[p]
            gr, gc = FL[target_loc]
            dists[p] = (abs(taxi_row - gr) + abs(taxi_col - gc)) / MD
    parts.append(dists)

    # GROUP 3: obstacle proximity, radius 2, 4 dirs (4)
    prox = np.zeros(4, dtype=np.float32)
    for d_idx, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):
        blocked = 0
        for step in range(1, OBS_RADIUS + 1):
            nr, nc = taxi_row + dr * step, taxi_col + dc * step
            if not (0 <= nr < R and 0 <= nc < C) or (nr, nc) in OC:
                blocked += 1
        prox[d_idx] = blocked / OBS_RADIUS
    parts.append(prox)

    # GROUP 4: progress fractions (3)
    n_del    = sum(p_delivered)
    n_intaxi = sum(p_locs[p] == L and not p_delivered[p] for p in range(P))
    n_wait   = P - n_del - n_intaxi
    parts.append(np.array([n_del / P, n_intaxi / P, n_wait / P],
                           dtype=np.float32))

    # GROUP 5: bias (1)
    parts.append(np.array([1.0], dtype=np.float32))

    phi = np.concatenate(parts)
    assert len(phi) == feature_dim(cfg), \
        f"dim mismatch: got {len(phi)}, expected {feature_dim(cfg)}"
    return phi


def features_from_env(env, cfg: EnvConfig) -> np.ndarray:
    return extract_features(
        env._taxi_row, env._taxi_col,
        env._p_loc, env._p_dest, env._p_delivered,
        cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ε schedules
# ─────────────────────────────────────────────────────────────────────────────

def linear_decay(episodes: int, eps_start=1.0, eps_end=0.05,
                 decay_frac=0.8) -> Callable:
    decay_ep = int(episodes * decay_frac)
    def schedule(ep: int) -> float:
        if ep < decay_ep:
            return eps_start - (eps_start - eps_end) * (ep / decay_ep)
        return eps_end
    return schedule


def exponential_decay(eps_start=1.0, eps_end=0.05,
                      decay_rate=0.999) -> Callable:
    def schedule(ep: int) -> float:
        return max(eps_end, eps_start * (decay_rate ** ep))
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_fa(
    env,
    cfg:               EnvConfig,
    episodes:          int             = 5_000,
    alpha:             float           = 0.01,
    gamma:             float           = 0.95,
    epsilon:           float           = 0.1,
    epsilon_schedule:  Callable | None = None,
    use_action_mask:   bool            = True,
    seed:              int             = 42,
    track_epsilon:     bool            = False,
    log_interval:      int             = 500,
    W_init:            np.ndarray | None = None,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    D = feature_dim(cfg)

    if W_init is not None:
        assert W_init.shape == (N_ACTIONS, D), \
            f"W_init shape {W_init.shape} != expected ({N_ACTIONS}, {D})"
        W = W_init.copy().astype(np.float64)
    else:
        rng = np.random.default_rng(seed)
        W   = rng.standard_normal((N_ACTIONS, D)) * 0.01

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
                action = int(valid[np.argmax(W[valid] @ phi)])

            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            phi_next = features_from_env(env, cfg)

            if done or truncated:
                target = reward
            else:
                next_mask  = info["action_mask"] if use_action_mask \
                             else np.ones(N_ACTIONS, dtype=np.int8)
                next_valid = np.nonzero(next_mask)[0]
                best_next  = float(np.max(W[next_valid] @ phi_next)) \
                             if len(next_valid) > 0 else 0.0
                target = reward + gamma * best_next

            td_error   = target - float(W[action] @ phi)
            W[action] += alpha * td_error * phi
            phi = phi_next

        episode_rewards.append(total_reward)

        if log_interval and (ep + 1) % log_interval == 0:
            recent = np.mean(episode_rewards[-log_interval:])
            print(f"  ep {ep+1:>6} | mean reward (last {log_interval}): "
                  f"{recent:+.1f}  ε={eps:.3f}")

    return {
        "W":               W,
        "episode_rewards": episode_rewards,
        "mean_reward":     float(np.mean(episode_rewards)),
        "std_reward":      float(np.std(episode_rewards)),
        "epsilons":        epsilon_trace if track_epsilon else None,
        "feature_dim":     D,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    env,
    cfg:             EnvConfig,
    W:               np.ndarray,
    n_episodes:      int  = 200,
    seed:            int  = 9999,
    use_action_mask: bool = True,
) -> dict:
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
            action = int(valid[np.argmax(W[valid] @ phi)])

            obs, reward, done, truncated, info = env.step(action)
            phi = features_from_env(env, cfg)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    env.close()
    return {
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_length":  float(np.mean(lengths)),
        "success_rate": float(np.mean([r > 0 for r in rewards])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature names (for weight plots)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_names(cfg: EnvConfig) -> list[str]:
    P     = cfg.n_passengers
    names = ["taxi_row", "taxi_col"]
    for p in range(P):
        names += [f"p{p}_goal_Δrow", f"p{p}_goal_Δcol"]
    for p in range(P):
        names.append(f"p{p}_goal_dist")
    names += ["obs_N", "obs_S", "obs_E", "obs_W"]
    names += ["frac_delivered", "frac_in_taxi", "frac_waiting", "bias"]
    return names


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
        f"Linear FA Q-learning — {cfg.n_passengers}p  "
        f"{cfg.grid_rows}×{cfg.grid_cols}  D={feature_dim(cfg)}"
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


def plot_weights(W_mean: np.ndarray, cfg: EnvConfig,
                 output_dir: Path, tag: str = "") -> None:
    names  = build_feature_names(cfg)
    labels = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    vmax   = np.abs(W_mean).max()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for a, ax in enumerate(axes.flat):
        w = W_mean[a]
        colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in w]
        ax.barh(range(len(w)), w, color=colors, height=0.7)
        ax.set_title(f"Action: {labels[a]}", fontsize=11)
        ax.set_xlim(-vmax * 1.1, vmax * 1.1)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle(
        f"Weight magnitudes per action" + (f"  [{tag}]" if tag else "") +
        "\n(green=positive, orange=negative)", fontsize=13
    )
    plt.tight_layout()
    fname = output_dir / (f"weights_{tag}.png" if tag else "weights.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main: Taxi-v3 (5×5) → SinglePassengerLargeTaxiEnv (15×15) transfer

# ─────────────────────────────────────────────────────────────────────────────
# Main: Taxi-v3 (5×5) → SinglePassengerLargeTaxiEnv (15×15) transfer
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from taxi_v3_wrapper import TaxiV3Wrapper
    from single_passenger_taxi_large import SinglePassengerLargeTaxiEnv

    output_dir  = Path("results/linear_fa")
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    N_RUNS    = 5
    BASE_SEED = 58922320
    seeds     = [BASE_SEED + i for i in range(N_RUNS)]

    # ── PHASE 1: train on Taxi-v3 (5×5) ─────────────────────────────────────
    EPISODES_SMALL = 10_000
    schedule_small = linear_decay(EPISODES_SMALL, eps_start=1.0,
                                  eps_end=0.05, decay_frac=0.95)

    _env_probe = TaxiV3Wrapper()
    cfg_small  = EnvConfig.from_env(_env_probe)
    _env_probe.close()

    print(f"Feature dim D = {feature_dim(cfg_small)}  "
          f"(same for both envs — transfer is direct)\n")
    print("=== PHASE 1: train on Taxi-v3 (5×5) ===")

    small_results = []
    small_eval_stats = []
    for run in range(N_RUNS):
        seed = seeds[run]
        print(f"─── Run {run+1}/{N_RUNS}  (seed {seed}) ───")
        env    = TaxiV3Wrapper()
        result = train_linear_fa(
            env, cfg_small,
            episodes         = EPISODES_SMALL,
            alpha            = 0.01,
            gamma            = 0.95,
            epsilon_schedule = schedule_small,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = True,
            log_interval     = 1000,
        )
        eval_stats = evaluate_policy(env, cfg_small, result["W"],
                                     n_episodes=500, seed=seed + 10000)
        print(f"  EVAL → mean: {eval_stats['mean_reward']:+.1f} ± "
              f"{eval_stats['std_reward']:.1f}  "
              f"success: {eval_stats['success_rate']:.1%}  "
              f"len: {eval_stats['mean_length']:.0f}\n")
        small_results.append(result)
        small_eval_stats.append(eval_stats)

    plot_training(small_results, cfg_small, output_dir, tag="taxiv3")
    W_small = np.mean([r["W"] for r in small_results], axis=0)
    np.save(output_dir / "W_taxiv3.npy", W_small)
    plot_weights(W_small, cfg_small, output_dir, tag="taxiv3")

    # per-run weights
    for run, result in enumerate(small_results):
        np.save(weights_dir / f"taxiv3_run{run+1}.npy", result["W"])

    # best-run checkpoint
    best_idx = max(range(N_RUNS), key=lambda i: small_results[i]["mean_reward"])
    np.savez(
        output_dir / "best_taxiv3.npz",
        W           = small_results[best_idx]["W"],
        mean_reward = small_results[best_idx]["mean_reward"],
        std_reward  = small_results[best_idx]["std_reward"],
        run_index   = best_idx,
        seed        = seeds[best_idx],
    )

    # summary.json — Phase 1
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

    # ── PHASE 2: zero-shot eval on 15×15 with Taxi-v3 weights ───────────────
    _env_probe = SinglePassengerLargeTaxiEnv()
    cfg_large  = EnvConfig.from_env(_env_probe)
    _env_probe.close()

    D_small = feature_dim(cfg_small)
    D_large = feature_dim(cfg_large)

    print(f"\n=== PHASE 2: transfer check ===")
    print(f"Taxi-v3 D={D_small}  |  15×15 D={D_large}")

    assert D_small == D_large, (
        f"Dim mismatch ({D_small} vs {D_large}) — "
        "both envs must have the same n_passengers."
    )

    print("Dims match — running zero-shot eval on 15×15 with Taxi-v3 weights.")
    env_zs   = SinglePassengerLargeTaxiEnv(max_steps=500)
    zs_stats = evaluate_policy(env_zs, cfg_large, W_small,
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

    for run in range(N_RUNS):
        seed = seeds[run]
        print(f"─── Run {run+1}/{N_RUNS}  (seed {seed}) ───")

        env_ft    = SinglePassengerLargeTaxiEnv(max_steps=500)
        result_ft = train_linear_fa(
            env_ft, cfg_large,
            episodes         = EPISODES_LARGE,
            alpha            = 0.005,
            gamma            = 0.95,
            epsilon_schedule = schedule_ft,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = True,
            log_interval     = 2000,
            W_init           = W_small,
        )
        eval_ft = evaluate_policy(env_ft, cfg_large, result_ft["W"],
                                  n_episodes=500, seed=seed + 20000)
        print(f"  Fine-tune EVAL → mean: {eval_ft['mean_reward']:+.1f}  "
              f"success: {eval_ft['success_rate']:.1%}")
        finetune_results.append(result_ft)
        finetune_eval.append(eval_ft)

        env_sc    = SinglePassengerLargeTaxiEnv(max_steps=500)
        result_sc = train_linear_fa(
            env_sc, cfg_large,
            episodes         = EPISODES_LARGE,
            alpha            = 0.005,
            gamma            = 0.95,
            epsilon_schedule = schedule_scratch,
            use_action_mask  = True,
            seed             = seed,
            track_epsilon    = False,
            log_interval     = 2000,
            W_init           = None,
        )
        eval_sc = evaluate_policy(env_sc, cfg_large, result_sc["W"],
                                  n_episodes=500, seed=seed + 20000)
        print(f"  Scratch   EVAL → mean: {eval_sc['mean_reward']:+.1f}  "
              f"success: {eval_sc['success_rate']:.1%}\n")
        scratch_results.append(result_sc)
        scratch_eval.append(eval_sc)

    # ── transfer comparison plot ──────────────────────────────────────────────
    def smooth(x, w=300):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ft_mean = np.mean([r["episode_rewards"] for r in finetune_results], axis=0)
    sc_mean = np.mean([r["episode_rewards"] for r in scratch_results],  axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(smooth(ft_mean), label="Fine-tune (Taxi-v3 → 15×15)", color="#1D9E75")
    plt.plot(smooth(sc_mean), label="Scratch (15×15)",              color="#D85A30")
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.title("Transfer learning: Taxi-v3 → 15×15  vs  from scratch")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    plot_training(finetune_results, cfg_large, output_dir, tag="15x15_finetune")
    W_large_ft = np.mean([r["W"] for r in finetune_results], axis=0)
    np.save(output_dir / "W_large_finetuned.npy", W_large_ft)
    np.save(output_dir / "W_large_scratch.npy",
            np.mean([r["W"] for r in scratch_results], axis=0))
    plot_weights(W_large_ft, cfg_large, output_dir, tag="15x15_finetuned")

    # per-run weights
    for run, (r_ft, r_sc) in enumerate(zip(finetune_results, scratch_results)):
        np.save(weights_dir / f"15x15_finetune_run{run+1}.npy", r_ft["W"])
        np.save(weights_dir / f"15x15_scratch_run{run+1}.npy",  r_sc["W"])

    # best-run checkpoints
    best_ft_idx = max(range(N_RUNS), key=lambda i: finetune_results[i]["mean_reward"])
    best_sc_idx = max(range(N_RUNS), key=lambda i: scratch_results[i]["mean_reward"])

    np.savez(
        output_dir / "best_15x15_finetune.npz",
        W           = finetune_results[best_ft_idx]["W"],
        mean_reward = finetune_results[best_ft_idx]["mean_reward"],
        std_reward  = finetune_results[best_ft_idx]["std_reward"],
        run_index   = best_ft_idx,
        seed        = seeds[best_ft_idx],
    )
    np.savez(
        output_dir / "best_15x15_scratch.npz",
        W           = scratch_results[best_sc_idx]["W"],
        mean_reward = scratch_results[best_sc_idx]["mean_reward"],
        std_reward  = scratch_results[best_sc_idx]["std_reward"],
        run_index   = best_sc_idx,
        seed        = seeds[best_sc_idx],
    )

    # summary.json — Phase 3
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
    print(f"  Plots         : training_taxiv3.png, weights_taxiv3.png,")
    print(f"                  training_15x15_finetune.png, weights_15x15_finetuned.png,")
    print(f"                  transfer_comparison.png")
    print(f"  Mean weights  : W_taxiv3.npy, W_large_finetuned.npy, W_large_scratch.npy")
    print(f"  Per-run       : weights/taxiv3_run{{1-{N_RUNS}}}.npy")
    print(f"                  weights/15x15_finetune_run{{1-{N_RUNS}}}.npy")
    print(f"                  weights/15x15_scratch_run{{1-{N_RUNS}}}.npy")
    print(f"  Best runs     : best_taxiv3.npz, best_15x15_finetune.npz, best_15x15_scratch.npz")
    print(f"  Summary       : summary.json")