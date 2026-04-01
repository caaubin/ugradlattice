"""
Shared utilities for LQCD Jupyter notebooks.

Keeps notebook cells short by centralizing:
- Path setup so `import su2` and Propagator modules work
- Config loading for both dict and list pickle formats
- Quick Metropolis wrapper for demo runs
- Standardized plotting helpers
"""

import sys
import os
import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def setup_paths():
    """Add su2/ and su2/meson_correlator/ to sys.path so notebooks can
    ``import su2`` and ``from Propagator import ...`` regardless of cwd."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    su2_dir = os.path.join(base, "su2")
    meson_dir = os.path.join(su2_dir, "meson_correlator")
    for p in (su2_dir, meson_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return base  # repo root, handy for locating configs/


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(filepath):
    """Load a gauge configuration pickle.

    Handles two on-disk formats produced by different scripts:
    * dict  ``{'U': array, 'plaquette': float, ...}``
    * list  ``[plaquette, U_array]``

    Returns
    -------
    U : ndarray
        Gauge field array, shape ``(V, 4, 4)``
    metadata : dict
        Any extra info (plaquette, beta, dims, ...).
    """
    import pickle
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        U = data.pop("U")
        return U, data
    elif isinstance(data, list) and len(data) >= 2:
        plaq, U = data[0], data[1]
        # Older scripts store U as (V*ndim, 4) -- reshape to (V, ndim, 4)
        if U.ndim == 2 and U.shape[1] == 4:
            ndim = 4
            V = U.shape[0] // ndim
            U = U.reshape(V, ndim, 4)
        return U, {"plaquette": plaq}
    else:
        raise ValueError(f"Unknown config format in {filepath}")


# ---------------------------------------------------------------------------
# Quick Monte Carlo
# ---------------------------------------------------------------------------

def quick_metropolis(La, beta, n_sweeps, start="cold", seed=None):
    """Run a short Metropolis simulation and return plaquette history.

    Parameters
    ----------
    La : list
        Lattice dimensions, e.g. ``[4, 4, 4, 4]``.
    beta : float
        Coupling constant.
    n_sweeps : int
        Number of full lattice sweeps.
    start : str
        ``'cold'`` (identity) or ``'hot'`` (random).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    plaquettes : ndarray of shape ``(n_sweeps,)``
    U : ndarray
        Final gauge field.
    """
    import su2

    if seed is not None:
        np.random.seed(seed)

    V = su2.vol(La)
    D = su2.dim(La)
    ndim = len(La)
    Mlink = 10  # hits per link

    # Gauge field
    U = np.zeros((V, ndim, 4))
    mups = np.zeros((V, ndim), dtype=int)
    mdns = np.zeros((V, ndim), dtype=int)

    init_fn = su2.cstart if start == "cold" else su2.hstart
    for i in range(V):
        for mu in range(ndim):
            U[i][mu] = init_fn()
            mups[i, mu] = su2.mupi(i, mu, La)
            mdns[i, mu] = su2.mdowni(i, mu, La)

    plaquettes = np.zeros(n_sweeps)

    report_every = max(1, n_sweeps // 10)

    for m in range(n_sweeps):
        for i in range(V):
            for mu in D:
                U0 = U[i][mu].copy()
                staples = su2.getstaple(U, i, mups, mdns, mu)
                for _ in range(Mlink):
                    U0n = su2.update(U0)
                    dS = -0.5 * su2.tr(su2.mult(U0n - U0, staples))
                    if dS < 0 or np.random.random() < np.exp(-beta * dS):
                        U[i][mu] = U0n
                        U0 = U0n
        plaquettes[m] = su2.calcPlaq(U, La, mups)
        if m == 0 or (m + 1) % report_every == 0:
            print(f"  Sweep {m+1}/{n_sweeps}: P = {plaquettes[m]:.6f}")

    return plaquettes, U


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_correlator(correlator, title="Meson Correlator", ax=None):
    """Semi-log plot of ``|C(t)|`` with optional axis."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(correlator))
    c = np.abs(np.asarray(correlator, dtype=float))
    ax.semilogy(t, np.maximum(c, 1e-30), "o-", markersize=5)
    ax.set_xlabel("t")
    ax.set_ylabel("|C(t)|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def autocorrelation(data, max_lag=None):
    """Compute normalized autocorrelation function.

    Parameters
    ----------
    data : array_like
        Time series (e.g. plaquette history from a Monte Carlo run).
    max_lag : int, optional
        Maximum lag to compute.  Defaults to ``len(data) // 4``.

    Returns
    -------
    rho : ndarray of shape ``(max_lag,)``
        Normalized autocorrelation: ``rho[0] = 1``.
    """
    x = np.asarray(data, dtype=float) - np.mean(data)
    if max_lag is None:
        max_lag = len(x) // 4
    c0 = np.dot(x, x)
    if c0 == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:len(x) - k], x[k:]) / c0
                     for k in range(max_lag)])


def plot_effective_mass(correlator, title="Effective Mass", ax=None):
    """Plot ``m_eff(t) = ln[C(t)/C(t+1)]`` with a simple plateau guide."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    c = np.abs(np.asarray(correlator, dtype=float))
    c = np.maximum(c, 1e-30)
    m_eff = np.log(c[:-1] / c[1:])
    t = np.arange(len(m_eff))

    valid = np.isfinite(m_eff) & (m_eff > 0) & (m_eff < 10)
    ax.errorbar(t[valid], m_eff[valid], fmt="s-", markersize=5, capsize=3)
    if np.any(valid):
        plateau = np.median(m_eff[valid])
        ax.axhline(plateau, ls="--", color="r", label=f"median = {plateau:.3f}")
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$m_{\rm eff}(t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
