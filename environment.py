"""
environment.py
--------------
FrozenLake-v1 environment utilities and exact value computation.

The FrozenLake environment models a stochastic navigation problem:
- 4x4 grid with Start (S), Frozen (F), Hole (H), and Goal (G) cells
- Slippery ice: each action moves the intended direction with p=1/3,
  or 90° left/right with p=1/3 each
- Reward: +1 on reaching Goal, 0 otherwise
- Episodes terminate on Hole or Goal

With γ=1, the value function equals the probability of reaching the
goal before falling into a hole — an intuitive success metric.
"""

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(seed=0, is_slippery=True, map_name="4x4"):
    """
    Create a FrozenLake-v1 environment without a renderer (fast training).

    Parameters
    ----------
    seed        : int   RNG seed
    is_slippery : bool  stochastic (True) or deterministic (False) transitions
    map_name    : str   '4x4' or '8x8'

    Returns
    -------
    env : gym.Env
    """
    env = gym.make("FrozenLake-v1", map_name=map_name,
                   is_slippery=is_slippery)
    env.reset(seed=seed)
    return env


def decode_map(env):
    """Return the grid layout as a list of lists of characters."""
    desc = env.unwrapped.desc
    return [[c.decode("utf-8") for c in row] for row in desc]


def get_absorbing_states(env):
    """Return (absorbing_set, goal_set) of state indices."""
    grid = decode_map(env)
    n    = len(grid)
    absorbing, goal_states = set(), set()
    for r in range(n):
        for c in range(n):
            idx = r * n + c
            ch  = grid[r][c]
            if ch in ("H", "G"):
                absorbing.add(idx)
            if ch == "G":
                goal_states.add(idx)
    return absorbing, goal_states


# ------------------------------------------------------------------
# Exact value computation — random policy, γ=1
# ------------------------------------------------------------------

def exact_value_random_policy(env):
    """
    Compute V^π exactly for the uniform-random policy at γ=1.

    Solves (I - Q) v = r over non-terminal states, where:
        Q[i,j] = sum_a (1/nA) * P(s_j | s_i, a)   for non-terminal s_i, s_j
        r[i]   = sum_a (1/nA) * P(goal | s_i, a)

    With γ=1, V^π(s) = P(reach goal before hole | start at s, follow π).

    Returns
    -------
    V : np.ndarray shape (nS,)  — terminals have value 0
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    absorbing, goal_states = get_absorbing_states(env)

    nonterm = [s for s in range(nS) if s not in absorbing]
    idx_map = {s: i for i, s in enumerate(nonterm)}
    m       = len(nonterm)

    Q_mat = np.zeros((m, m))
    r_vec = np.zeros(m)
    p_act = 1.0 / nA

    for s in nonterm:
        i = idx_map[s]
        for a in range(nA):
            for (p, ns, rew, done) in env.unwrapped.P[s][a]:
                if ns in absorbing:
                    r_vec[i] += p_act * p * (1.0 if ns in goal_states else 0.0)
                else:
                    Q_mat[i, idx_map[ns]] += p_act * p

    v = np.linalg.solve(np.eye(m) - Q_mat, r_vec)

    V = np.zeros(nS)
    for s in nonterm:
        V[s] = v[idx_map[s]]
    return V


# ------------------------------------------------------------------
# Exact value computation — deterministic policy, γ=1
# ------------------------------------------------------------------

def exact_value_deterministic_policy(env, pi):
    """
    Compute V^π exactly for a deterministic policy at γ=1.

    Parameters
    ----------
    pi : np.ndarray shape (nS,) — action index per state

    Returns
    -------
    V : np.ndarray shape (nS,)
    """
    nS = env.observation_space.n
    absorbing, goal_states = get_absorbing_states(env)

    nonterm = [s for s in range(nS) if s not in absorbing]
    idx_map = {s: i for i, s in enumerate(nonterm)}
    m       = len(nonterm)

    Q_mat = np.zeros((m, m))
    r_vec = np.zeros(m)

    for s in nonterm:
        a = int(pi[s])
        i = idx_map[s]
        for (p, ns, rew, done) in env.unwrapped.P[s][a]:
            if ns in absorbing:
                r_vec[i] += p * (1.0 if ns in goal_states else 0.0)
            else:
                Q_mat[i, idx_map[ns]] += p

    v = np.linalg.solve(np.eye(m) - Q_mat, r_vec)

    V = np.zeros(nS)
    for s in nonterm:
        V[s] = v[idx_map[s]]
    return V


# ------------------------------------------------------------------
# Episode simulation
# ------------------------------------------------------------------

def run_episode(env, policy, max_steps=400, seed=None):
    """
    Roll out one episode.

    Parameters
    ----------
    policy : callable  state -> action

    Returns
    -------
    states, actions, rewards : lists
    """
    if seed is not None:
        s, _ = env.reset(seed=seed)
    else:
        s, _ = env.reset()
    states, actions, rewards = [], [], []
    done = False
    while not done and len(states) < max_steps:
        a = policy(s)
        s2, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        s    = s2
        done = terminated or truncated
    return states, actions, rewards


def random_policy(env):
    """Return a callable that samples uniformly from the action space."""
    return lambda _s: env.action_space.sample()
