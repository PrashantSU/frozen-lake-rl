"""
visualize.py
------------
Plotting utilities for the FrozenLake RL experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CELL_COLORS = {'S': '#3498db', 'F': '#ecf0f1', 'H': '#e74c3c', 'G': '#2ecc71'}


def plot_value_grid(V, title, env, ax=None, save_path=None):
    """
    Heatmap of a 4x4 value function with cell type annotations.

    Parameters
    ----------
    V    : np.ndarray shape (16,)
    env  : gym.Env  (used to read map layout)
    """
    from environment import decode_map
    grid = decode_map(env)
    n    = len(grid)
    G    = V.reshape(n, n)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))

    im = ax.imshow(G, cmap='YlGn', vmin=0, vmax=1, origin='upper')
    for r in range(n):
        for c in range(n):
            ch    = grid[r][c]
            color = 'white' if ch == 'H' else 'black'
            label = f"{G[r,c]:.2f}\n({ch})"
            ax.text(c, r, label, ha='center', va='center',
                    fontsize=8.5, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    if standalone:
        plt.colorbar(im, ax=ax, label='V(s) = P(reach goal)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    return im


def plot_mse_comparison(results, title, log_scale=False, save_path=None):
    """
    MSE vs episodes curves with ±1 std bands.

    Parameters
    ----------
    results : list of dicts  {'label', 'eps', 'mean', 'std'}
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for res in results:
        y = np.maximum(res['mean'], 1e-12) if log_scale else res['mean']
        ax.plot(res['eps'], y, label=res['label'])
        if not log_scale:
            ax.fill_between(res['eps'],
                            res['mean'] - res['std'],
                            res['mean'] + res['std'], alpha=0.18)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('MSE (log scale)')
    else:
        ax.axhline(0, ls='--', lw=0.8, color='gray')
        ax.set_ylabel('MSE vs true V^π')
    ax.set_xlabel('Episodes')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_success_curves(logs_dict, title, smooth_k=5, save_path=None):
    """
    Success rate vs training episodes for multiple algorithms.

    Parameters
    ----------
    logs_dict : dict  {label: {'episodes': array, 'success': array}}
    smooth_k  : int   moving-average window
    """
    from control import moving_average
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors  = {'SARSA': '#2980b9', 'Q-learning': '#27ae60'}
    for label, log in logs_dict.items():
        smoothed = moving_average(log['success'], k=smooth_k)
        ax.plot(log['episodes'], smoothed,
                label=label, color=colors.get(label))
        ax.plot(log['episodes'], log['success'],
                alpha=0.25, color=colors.get(label))
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, ls='--', lw=0.8, color='gray')
    ax.set_xlabel('Training episodes')
    ax.set_ylabel('Success rate (greedy evaluation)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_offline_convergence(start_value_hist, V_opt_start=None, save_path=None):
    """
    True start-state value of greedy policy over offline training epochs.
    """
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(start_value_hist, 'o-', color='steelblue', label='V^π(start) — greedy policy')
    if V_opt_start is not None:
        ax.axhline(V_opt_start, ls='--', color='tomato', lw=1.5,
                   label=f'True V*(start) = {V_opt_start:.3f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('V^π(start state)')
    ax.set_title('Offline Q-learning: convergence of greedy policy value', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_snapshot_grid(snapshots, env, ncols=4, save_path=None):
    """
    Grid of value function heatmaps at different training episodes.
    """
    n     = len(snapshots)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, (ep, V) in enumerate(snapshots):
        title = "Init (ep 0)" if ep == 0 else f"Episode {ep}"
        im = plot_value_grid(V, title, env, ax=axes[i])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
