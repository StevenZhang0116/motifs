"""Mixed trace motif moments and cumulants.

The mixed trace family in Recanatesi et al. (2019) measures two directed paths
that share both their starting and ending neurons. Paper:
https://doi.org/10.1371/journal.pcbi.1006446
Supplementary equations: https://doi.org/10.1371/journal.pcbi.1006446.s001

It generalizes familiar
closed motifs: for example, the ``(2, 1)`` moment detects a two-step path and a
direct edge between the same ordered pair, i.e. a weighted feed-forward loop.

Adjacency convention
--------------------
``W[i, j]`` is the weight of the directed edge ``j -> i``.

For ``n, m >= 1``, the paper-normalized moment is

    mu_trace[n,m]
        = Tr(W**n @ (W.T)**m) / N**(n + m + 1).

Let ``u = ones(N) / sqrt(N)``, ``Theta = I - u @ u.T``, and
``W_n_theta = (W @ Theta)**(n - 1) @ W``.  The corresponding cumulant is

    kappa_trace[n,m]
        = Tr(W_n_theta @ Theta @ W_m_theta.T @ Theta)
          / N**(n + m + 1).

The PLOS paper uses the extra factor ``1/N`` because these trace terms later
appear multiplied by ``N`` in covariance-trace formulas.  Set
``normalization='cycle_compatible'`` to multiply both moments and cumulants by
``N`` so their scaling matches :mod:`motif_cumulants.cycle`.

This two-index path-pair trace family is not the one-index closed-walk cycle
family in :mod:`motif_cumulants.cycle`, and it is not the binary SONET
reciprocal excess in :mod:`motif_cumulants.second_order`.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypedDict

import numpy as np

from ._validation import (
    is_sparse_matrix,
    prepare_adjacency,
    validate_max_order,
)


TraceNormalization = Literal["recanatesi", "cycle_compatible"]


class MixedTraceMotifResult(TypedDict, total=False):
    """Dictionary returned by :func:`mixed_trace_motif_cumulants`."""

    path_order: np.ndarray
    total_order: np.ndarray
    cumulants: np.ndarray
    moments: np.ndarray
    normalization_factor: float


def _validate_normalization(normalization: str) -> TraceNormalization:
    if normalization not in ("recanatesi", "cycle_compatible"):
        raise ValueError(
            "normalization must be 'recanatesi' or 'cycle_compatible'."
        )
    return normalization  # type: ignore[return-value]


def _normalization_factor(
    normalization: TraceNormalization,
    n_nodes: int,
) -> float:
    return 1.0 / n_nodes if normalization == "recanatesi" else 1.0


def _frobenius_inner(left: Any, right: Any) -> float:
    """Return the real Frobenius inner product without unnecessary densifying."""
    if is_sparse_matrix(left) and is_sparse_matrix(right):
        return float(left.multiply(right).sum())
    return float(np.sum(np.asarray(left) * np.asarray(right)))


def _mixed_trace_moments_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    *,
    normalization: TraceNormalization,
) -> np.ndarray:
    scaled = matrix / n_nodes
    powers: list[Any] = []
    power = scaled.copy()

    for index in range(max_order):
        powers.append(power)
        if index + 1 < max_order:
            power = power @ scaled

    factor = _normalization_factor(normalization, n_nodes)
    moments = np.empty((max_order, max_order), dtype=np.float64)

    for n_index in range(max_order):
        for m_index in range(n_index, max_order):
            value = factor * _frobenius_inner(
                powers[n_index],
                powers[m_index],
            )
            moments[n_index, m_index] = value
            moments[m_index, n_index] = value

    return moments


def mixed_trace_motif_moments(
    W: Any,
    max_order: int,
    *,
    normalization: TraceNormalization = "recanatesi",
) -> np.ndarray:
    """Calculate mixed trace moments for path lengths ``1..max_order``.

    The entry ``[n - 1, m - 1]`` is based on
    ``Tr(W**n @ (W.T)**m)``.  With the default paper normalization it is
    divided by ``N**(n + m + 1)``.
    """
    max_order = validate_max_order(max_order)
    normalization = _validate_normalization(normalization)
    matrix, n_nodes = prepare_adjacency(W)
    return _mixed_trace_moments_from_prepared_matrix(
        matrix,
        n_nodes,
        max_order,
        normalization=normalization,
    )


def _theta_left(matrix: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply ``Theta`` on the left without forming the projector."""
    return matrix - np.outer(u, u @ matrix)


def _theta_right(matrix: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply ``Theta`` on the right without forming the projector."""
    return matrix - np.outer(matrix @ u, u)


def _mixed_trace_cumulants_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    *,
    normalization: TraceNormalization,
) -> np.ndarray:
    if is_sparse_matrix(matrix):
        warnings.warn(
            "Mixed trace cumulants contain the dense projector Theta and "
            "therefore convert a sparse adjacency matrix to dense.",
            RuntimeWarning,
            stacklevel=3,
        )
        scaled = matrix.toarray() / n_nodes
    else:
        scaled = np.array(matrix / n_nodes, dtype=np.float64, copy=True)

    u = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)
    projected_basis: list[np.ndarray] = []

    # B_n = (A Theta)^(n-1) A, where A = W/N.
    basis = scaled.copy()
    for index in range(max_order):
        doubly_projected = _theta_right(_theta_left(basis, u), u)
        projected_basis.append(doubly_projected)

        if index + 1 < max_order:
            basis = scaled @ _theta_left(basis, u)

    factor = _normalization_factor(normalization, n_nodes)
    cumulants = np.empty((max_order, max_order), dtype=np.float64)

    # Tr(B_n Theta B_m^T Theta)
    #     = <Theta B_n Theta, Theta B_m Theta>_F.
    for n_index in range(max_order):
        for m_index in range(n_index, max_order):
            value = factor * float(
                np.sum(
                    projected_basis[n_index]
                    * projected_basis[m_index]
                )
            )
            cumulants[n_index, m_index] = value
            cumulants[m_index, n_index] = value

    return cumulants


def mixed_trace_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    normalization: TraceNormalization = "recanatesi",
    return_moments: bool = True,
) -> MixedTraceMotifResult:
    """Calculate mixed trace motif cumulants for all path-length pairs.

    ``result["cumulants"][n - 1, m - 1]`` is the irreducible mixed trace
    statistic for two paths of lengths ``n`` and ``m`` that share both
    endpoints.
    """
    max_order = validate_max_order(max_order)
    normalization = _validate_normalization(normalization)
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    factor = _normalization_factor(normalization, n_nodes)
    cumulants = _mixed_trace_cumulants_from_prepared_matrix(
        matrix,
        n_nodes,
        max_order,
        normalization=normalization,
    )

    orders = np.arange(1, max_order + 1, dtype=int)
    result: MixedTraceMotifResult = {
        "path_order": orders,
        "total_order": np.add.outer(orders, orders),
        "cumulants": cumulants,
        "normalization_factor": factor,
    }

    if return_moments:
        result["moments"] = _mixed_trace_moments_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
            normalization=normalization,
        )

    return result
