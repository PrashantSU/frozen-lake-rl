"""
main.py
-------
Reproduces all four experiments:

  1. Exact value function of the random policy (analytical)
  2. MC vs TD(0) prediction — MSE convergence comparison
  3. SARSA vs Q-learning control — success rate curves
  4. Offline Q-learning from random behavioral data

Run:
    python main.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import (make_env, exact_value_random_policy,
                         exact_value_deterministic_policy, random_policy)
from prediction  import mc_first_visit, td0, mse_curve, aggregate_trials
from control     import (train_sarsa, train_qlearning,
                         collect_random_dataset, offline_qlearning,
                         moving_average)
from visualize   import (plot_value_grid, plot_mse_comparison,
                         plot_success_curves, plot_offline_convergence,
                         plot_snapshot_grid)

os.makedirs("results", exist_ok=True)
SEED = 0

# ======================================================================
# 1. Exact value — random policy, γ=1
# ======================================================================
print("=== 1. Exact value function (random policy, γ=1) ===")
env = make_env(seed=SEED)
V_true = exact_value_random_policy(env)

print(f"  V(start=0) = {V_true[0]:.4f}  "
      f"(random policy reaches goal with ~{V_true[0]*100:.1f}% probability)")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
im = plot_value_grid(V_true, "True V^π — Random Policy (γ=1)", env, ax=ax)
plt.colorbar(im, ax=ax, label='P(reach goal)')
plt.tight_layout()
plt.savefig("results/true_value_random_policy.png", dpi=150, bbox_inches='tight')
plt.show()

# ======================================================================
# 2. MC vs TD(0) — MSE convergence
# ======================================================================
print("\n=== 2. MC vs TD(0) prediction ===")

TRIALS        = 5
EPISODES      = 8000
SNAPSHOT_EVERY= 400
GAMMA         = 1.0

TD_CONFIGS = [
    {"label": "TD const α=0.20",   "schedule": "constant",     "alpha0": 0.20},
    {"label": "TD const α=0.05",   "schedule": "constant",     "alpha0": 0.05},
    {"label": "TD 1/(1+N)",        "schedule": "inverse",      "alpha0": 1.0},
    {"label": "TD 1/√(1+N)",       "schedule": "inverse_sqrt", "alpha0": 1.0},
]

def run_mc_trials(trials, episodes, snapshot_every, gamma, seed_offset=100):
    eps_list, mse_list = [], []
    for k in range(trials):
        e = make_env(seed=seed_offset + k)
        snaps = mc_first_visit(e, random_policy(e), gamma=gamma,
                               episodes=episodes, snapshot_every=snapshot_every)
        ep, ms = mse_curve(snaps, V_true)
        eps_list.append(ep); mse_list.append(ms)
    return aggregate_trials(eps_list, mse_list)

mc_eps, mc_mean, mc_std = run_mc_trials(TRIALS, EPISODES, SNAPSHOT_EVERY, GAMMA)
mc_result = {"label": "First-Visit MC", "eps": mc_eps,
             "mean": mc_mean, "std": mc_std}

td_results = [mc_result]
for i, cfg in enumerate(TD_CONFIGS):
    eps_list, mse_list = [], []
    for k in range(TRIALS):
        e = make_env(seed=200 + 100 * i + k)
        snaps = td0(e, random_policy(e), gamma=GAMMA,
                    alpha0=cfg["alpha0"], episodes=EPISODES,
                    snapshot_every=SNAPSHOT_EVERY, schedule=cfg["schedule"])
        ep, ms = mse_curve(snaps, V_true)
        eps_list.append(ep); mse_list.append(ms)
    ep, mean, std = aggregate_trials(eps_list, mse_list)
    td_results.append({"label": cfg["label"], "eps": ep,
                       "mean": mean, "std": std})
    print(f"  {cfg['label']:<25} final MSE = {mean[-1]:.5f}")

print(f"  {'First-Visit MC':<25} final MSE = {mc_mean[-1]:.5f}")

plot_mse_comparison(td_results,
    "MC vs TD(0): MSE convergence — FrozenLake 4×4 (γ=1)",
    log_scale=False,
    save_path="results/mse_linear.png")

plot_mse_comparison(td_results,
    "MC vs TD(0): MSE convergence — log scale",
    log_scale=True,
    save_path="results/mse_log.png")

# ======================================================================
# 3. SARSA vs Q-learning
# ======================================================================
print("\n=== 3. SARSA vs Q-learning ===")

env3 = make_env(seed=123)
Q_sarsa, log_sarsa = train_sarsa(env3, episodes=10_000, alpha=0.1,
                                 epsilon=0.1, gamma=1.0, seed=2, log_every=50)
Q_ql,    log_ql    = train_qlearning(env3, episodes=10_000, alpha=0.1,
                                     epsilon=0.1, gamma=1.0, seed=3, log_every=50)

print(f"  SARSA final success rate:      {log_sarsa['success'][-1]:.3f}")
print(f"  Q-learning final success rate: {log_ql['success'][-1]:.3f}")

plot_success_curves(
    {"SARSA": log_sarsa, "Q-learning": log_ql},
    "SARSA vs Q-learning — FrozenLake 4×4, slippery (γ=1)",
    smooth_k=5,
    save_path="results/sarsa_vs_qlearning.png"
)

# ======================================================================
# 4. Offline Q-learning
# ======================================================================
print("\n=== 4. Offline Q-learning (batch, off-policy) ===")

env4   = make_env(seed=2024)
D      = collect_random_dataset(env4, episodes=1000, seed=42)
Q_off, trace = offline_qlearning(env4, D, epochs=25, alpha=0.1,
                                 gamma=1.0, seed=7)

pi_final  = np.argmax(Q_off, axis=1).astype(int)
V_pi_final= exact_value_deterministic_policy(env4, pi_final)

print(f"  Dataset size: {len(D)} transitions")
print(f"  Final greedy policy V(start) = {V_pi_final[0]:.4f}")
print(f"  True V*(start) ≈ {max(trace['start_value_hist']):.4f}")

plot_offline_convergence(
    trace["start_value_hist"],
    V_opt_start=None,
    save_path="results/offline_convergence.png"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
im1 = plot_value_grid(V_true,    "True V^π — Random Policy",        env4, ax=axes[0])
im2 = plot_value_grid(V_pi_final,"True V^π — Offline Greedy Policy",env4, ax=axes[1])
for im, ax in zip([im1, im2], axes):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Offline Q-learning: random policy vs learned greedy policy", fontsize=12)
plt.tight_layout()
plt.savefig("results/offline_value_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nAll results saved to results/")
