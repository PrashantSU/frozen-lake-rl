"""
control.py
----------
Tabular control algorithms for FrozenLake-v1:

  - SARSA        (on-policy,  ε-greedy)
  - Q-learning   (off-policy, ε-greedy behavior)
  - Offline Q-learning from a fixed random-policy dataset

All algorithms use random tie-breaking among greedy actions to avoid
systematic bias toward action index 0 during early training.
"""

import numpy as np
from environment import exact_value_deterministic_policy


# ------------------------------------------------------------------
# Action selection helpers
# ------------------------------------------------------------------

def argmax_random_tie(q_values, rng):
    """Return the index of the maximum, breaking ties uniformly at random."""
    max_val    = np.max(q_values)
    candidates = np.flatnonzero(q_values == max_val)
    return int(rng.choice(candidates))


def epsilon_greedy(Q, s, epsilon, nA, rng):
    """
    ε-greedy action selection with random tie-breaking.

    With probability ε, select a random action.
    Otherwise, select greedily from Q[s] with random tie-breaking.
    """
    if rng.rand() < epsilon:
        return int(rng.randint(nA))
    return argmax_random_tie(Q[s], rng)


def evaluate_greedy(env, Q, episodes=300, seed=None):
    """
    Evaluate the greedy policy (ε=0) derived from Q.

    Returns the fraction of episodes that reach the goal.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    nA  = env.action_space.n
    successes = 0
    for _ in range(episodes):
        s, _ = env.reset()
        done  = False
        while not done:
            a = argmax_random_tie(Q[s], rng)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if done and r == 1.0:
                successes += 1
    return successes / episodes


# ------------------------------------------------------------------
# SARSA (on-policy)
# ------------------------------------------------------------------

def train_sarsa(env, episodes=10_000, alpha=0.1, epsilon=0.1,
                gamma=1.0, seed=0, log_every=50):
    """
    SARSA: on-policy temporal-difference control.

    Update rule:
        Q(s,a) += α [r + γ Q(s',a') - Q(s,a)]

    where a' is the actual next action sampled from the ε-greedy policy
    (on-policy). This makes SARSA sensitive to the exploration policy —
    it learns the value of the ε-greedy behaviour, not the greedy policy.

    Parameters
    ----------
    log_every : int  evaluate greedy success rate every N episodes

    Returns
    -------
    Q    : np.ndarray shape (nS, nA)
    logs : dict  {'episodes': array, 'success': array}
    """
    rng    = np.random.RandomState(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q      = np.zeros((nS, nA))
    ep_log, sr_log = [], []

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        a    = epsilon_greedy(Q, s, epsilon, nA, rng)
        done  = False

        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            a2      = epsilon_greedy(Q, s2, epsilon, nA, rng)
            target  = r + gamma * Q[s2, a2]      # on-policy target
            Q[s, a] += alpha * (target - Q[s, a])
            s, a    = s2, a2

        if ep % log_every == 0:
            ep_log.append(ep)
            sr_log.append(evaluate_greedy(env, Q, episodes=300))

    return Q, {"episodes": np.array(ep_log), "success": np.array(sr_log)}


# ------------------------------------------------------------------
# Q-learning (off-policy)
# ------------------------------------------------------------------

def train_qlearning(env, episodes=10_000, alpha=0.1, epsilon=0.1,
                    gamma=1.0, seed=1, log_every=50):
    """
    Q-learning: off-policy temporal-difference control.

    Update rule:
        Q(s,a) += α [r + γ max_{a'} Q(s',a') - Q(s,a)]

    The target uses the greedy value max Q(s',·), regardless of what
    action was actually taken next. This makes Q-learning off-policy:
    it directly learns the optimal action-value function Q*, even when
    behaviour is exploratory (ε-greedy).

    Returns
    -------
    Q    : np.ndarray shape (nS, nA)
    logs : dict  {'episodes': array, 'success': array}
    """
    rng    = np.random.RandomState(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q      = np.zeros((nS, nA))
    ep_log, sr_log = [], []

    for ep in range(1, episodes + 1):
        s, _  = env.reset()
        done   = False

        while not done:
            a       = epsilon_greedy(Q, s, epsilon, nA, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            done    = terminated or truncated
            target  = r if done else (r + gamma * np.max(Q[s2]))   # greedy target
            Q[s, a] += alpha * (target - Q[s, a])
            s       = s2

        if ep % log_every == 0:
            ep_log.append(ep)
            sr_log.append(evaluate_greedy(env, Q, episodes=300))

    return Q, {"episodes": np.array(ep_log), "success": np.array(sr_log)}


# ------------------------------------------------------------------
# Offline Q-learning (batch, off-policy)
# ------------------------------------------------------------------

def collect_random_dataset(env, episodes=1000, max_steps=400, seed=42):
    """
    Collect a fixed dataset of (s, a, r, s', done) transitions using
    a uniform-random behavioral policy — no model knowledge required.

    Returns
    -------
    D : list of (s, a, r, s', done) tuples
    """
    rng = np.random.RandomState(seed)
    nA  = env.action_space.n
    D   = []
    for _ in range(episodes):
        s, _  = env.reset()
        done   = False
        steps  = 0
        while not done and steps < max_steps:
            a = rng.randint(nA)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            D.append((s, a, float(r), s2, bool(done)))
            s     = s2
            steps += 1
    return D


def offline_qlearning(env, dataset, epochs=25, alpha=0.1, gamma=1.0, seed=7):
    """
    Offline (batch) Q-learning over a fixed dataset.

    The agent never interacts with the environment during training —
    it learns entirely from pre-collected transitions. This demonstrates
    the off-policy nature of Q-learning: the behavioral policy (random)
    differs from the target policy (greedy).

    Each epoch shuffles and replays the entire dataset once.
    After each epoch, the true start-state value of the current greedy
    policy is computed analytically (using the MDP model) to track convergence.

    Parameters
    ----------
    dataset : list of (s, a, r, s', done)
    epochs  : int  number of full passes over the dataset

    Returns
    -------
    Q     : np.ndarray shape (nS, nA)
    trace : dict  {'start_value_hist': list of float}
    """
    rng    = np.random.RandomState(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q      = np.zeros((nS, nA))

    S  = np.array([t[0] for t in dataset], dtype=int)
    A  = np.array([t[1] for t in dataset], dtype=int)
    R  = np.array([t[2] for t in dataset], dtype=float)
    NS = np.array([t[3] for t in dataset], dtype=int)
    DN = np.array([t[4] for t in dataset], dtype=bool)
    idx = np.arange(len(dataset))

    start_value_hist = []

    for _ in range(epochs):
        rng.shuffle(idx)
        for k in idx:
            s, a, r, ns, done = S[k], A[k], R[k], NS[k], DN[k]
            target  = r if done else (r + gamma * np.max(Q[ns]))
            Q[s, a] += alpha * (target - Q[s, a])

        pi    = np.argmax(Q, axis=1).astype(int)
        V_pi  = exact_value_deterministic_policy(env, pi)
        start_value_hist.append(float(V_pi[0]))

    return Q, {"start_value_hist": start_value_hist}


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def moving_average(x, k=10):
    """Smooth a 1D array with a simple moving average of window k."""
    k    = max(1, int(k))
    x    = np.asarray(x, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    core = (csum[k:] - csum[:-k]) / k
    y    = np.empty_like(x)
    y[:k-1] = core[0]
    y[k-1:] = core
    return y
