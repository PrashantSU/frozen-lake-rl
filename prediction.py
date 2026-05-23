"""
prediction.py
-------------
Value prediction algorithms for a fixed policy:

  - First-Visit Monte Carlo
  - TD(0) with configurable step-size schedules

Both methods estimate V^π(s) — the probability of reaching the goal
from state s under policy π (with γ=1 in FrozenLake).
"""

import numpy as np
from environment import run_episode


# ------------------------------------------------------------------
# First-Visit Monte Carlo
# ------------------------------------------------------------------

def mc_first_visit(env, policy, gamma=1.0, episodes=5000, snapshot_every=500):
    """
    First-Visit Monte Carlo prediction.

    For each episode, the return G_t is computed backwards. Each state's
    value is estimated as the running mean of returns from its first visit
    per episode. Updates occur only at episode end.

    With γ=1 in FrozenLake, G_t ∈ {0, 1} and V(s) → P(reach goal | s, π).

    Parameters
    ----------
    env            : gym.Env
    policy         : callable state -> action
    gamma          : float  discount factor
    episodes       : int    training episodes
    snapshot_every : int    snapshot interval

    Returns
    -------
    snapshots : list[(episode_idx, V_array)]  — includes (0, V_init)
    """
    nS = env.observation_space.n
    V            = np.zeros(nS)
    returns_sum  = np.zeros(nS)
    returns_count= np.zeros(nS)
    snapshots    = [(0, V.copy())]

    for ep in range(1, episodes + 1):
        states, actions, rewards = run_episode(env, policy)
        G, visited = 0.0, set()
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            s = states[t]
            if s not in visited:
                visited.add(s)
                returns_sum[s]   += G
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]
        if ep % snapshot_every == 0:
            snapshots.append((ep, V.copy()))

    return snapshots


# ------------------------------------------------------------------
# TD(0)
# ------------------------------------------------------------------

def td0(env, policy, gamma=1.0, alpha0=0.1, episodes=5000,
        snapshot_every=500, schedule="inverse_sqrt"):
    """
    TD(0) prediction with configurable step-size schedule.

    Online update after each step:
        target = r           if terminal
               = r + γ V(s') otherwise
        V(s)  += α_t(s) * (target - V(s))

    Step-size schedules
    -------------------
    'constant'     : α_t(s) = alpha0
    'inverse'      : α_t(s) = alpha0 / (1 + N_t(s))
    'inverse_sqrt' : α_t(s) = alpha0 / sqrt(1 + N_t(s))

    where N_t(s) is the visit count for state s up to step t.

    Parameters
    ----------
    schedule : str  one of {'constant', 'inverse', 'inverse_sqrt'}

    Returns
    -------
    snapshots : list[(episode_idx, V_array)]  — includes (0, V_init)
    """
    valid = {"constant", "inverse", "inverse_sqrt"}
    if schedule not in valid:
        raise ValueError(f"schedule must be one of {valid}, got {schedule!r}")

    nS     = env.observation_space.n
    V      = np.zeros(nS)
    visits = np.zeros(nS)
    snapshots = [(0, V.copy())]

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        done  = False
        while not done:
            a = policy(s)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            visits[s] += 1
            if schedule == "constant":
                alpha = alpha0
            elif schedule == "inverse":
                alpha = alpha0 / (1.0 + visits[s])
            else:  # inverse_sqrt
                alpha = alpha0 / np.sqrt(1.0 + visits[s])

            target = r if done else (r + gamma * V[s2])
            V[s]  += alpha * (target - V[s])
            s      = s2

        if ep % snapshot_every == 0:
            snapshots.append((ep, V.copy()))

    return snapshots


# ------------------------------------------------------------------
# MSE helpers
# ------------------------------------------------------------------

def mse_curve(snapshots, V_true):
    """Compute MSE vs V_true for each snapshot."""
    eps  = np.array([ep for ep, _ in snapshots], dtype=int)
    mses = np.array([np.mean((V - V_true) ** 2) for _, V in snapshots])
    return eps, mses


def aggregate_trials(eps_list, mse_list):
    """Mean and std of MSE curves across trials."""
    M    = np.stack(mse_list, axis=0)
    mean = M.mean(axis=0)
    std  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else M.std(axis=0)
    return eps_list[0], mean, std
