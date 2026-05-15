"""Generalization kernel and hypothesis simulators.

The master equation:

    Δy_{i,s} = (γ + δ R_i) · Σ_{s' ∈ A_i} ℓ_{i,s'} · exp(-d(s,s') / λ) + ε_{i,s}

Hypotheses:
- H1 (global gain):   Δy_{i,s} = α + η R_i + ε
- H2 (structured):    full equation (gain scales with R_i)
- H3 (widening):      λ → λ₀ + λ₁ R_i  (radius scales with R_i)
- H2+H3:              average of H2 and H3 (used for the comparison panel)
"""

from __future__ import annotations

import numpy as np


def kernel(d: np.ndarray, lam: float) -> np.ndarray:
    """k(d) = exp(-d / λ). Vectorised over arrays of any shape."""
    return np.exp(-np.asarray(d, dtype=float) / float(lam))


def generalization_field(
    source_idx: np.ndarray,
    mastery: np.ndarray,
    distance_matrix: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return length-n vector of Σ_{s'∈A} ℓ_{s'} · k(d(s,s')).

    distance_matrix[s, s'] is the distance from scene s to scene s'.
    """
    if len(source_idx) == 0:
        return np.zeros(distance_matrix.shape[0], dtype=float)
    d_to_sources = distance_matrix[:, source_idx]          # (n, |A|)
    ell = mastery[source_idx]                              # (|A|,)
    return (kernel(d_to_sources, lam) * ell[None, :]).sum(axis=1)


def simulate_h1(
    R: float,
    n_scenes: int,
    alpha: float,
    eta: float,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = alpha + eta * R
    return base + rng.normal(0.0, sigma, size=n_scenes)


def simulate_h2(
    R: float,
    source_idx: np.ndarray,
    mastery: np.ndarray,
    distance_matrix: np.ndarray,
    gamma: float,
    delta: float,
    lam: float,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = generalization_field(source_idx, mastery, distance_matrix, lam)
    return (gamma + delta * R) * field + rng.normal(0.0, sigma, size=len(field))


def simulate_h3(
    R: float,
    source_idx: np.ndarray,
    mastery: np.ndarray,
    distance_matrix: np.ndarray,
    gamma: float,
    lambda0: float,
    lambda1: float,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam_eff = lambda0 + lambda1 * R
    field = generalization_field(source_idx, mastery, distance_matrix, lam_eff)
    return gamma * field + rng.normal(0.0, sigma, size=len(field))


def simulate_h2h3(
    R: float,
    source_idx: np.ndarray,
    mastery: np.ndarray,
    distance_matrix: np.ndarray,
    gamma: float,
    delta: float,
    lambda0: float,
    lambda1: float,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam_eff = lambda0 + lambda1 * R
    field = generalization_field(source_idx, mastery, distance_matrix, lam_eff)
    return (gamma + delta * R) * field + rng.normal(0.0, sigma, size=len(field))
