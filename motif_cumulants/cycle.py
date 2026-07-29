"""Cycle motif moments and cumulants for directed weighted networks.

The definitions follow Hu et al., Physical Review E 98, 062312 (2018).
https://doi.org/10.1103/PhysRevE.98.062312
For an ``N x N`` weighted adjacency matrix ``W``, the order-``n`` cycle motif
moment is

    mu_cycle_n = N**(-n) * trace(W**n).

Let ``u = ones(N) / sqrt(N)`` and ``Theta = I - u @ u.T``.  The equivalent
direct matrix expression for the cycle motif cumulant is

    kappa_cycle_n = N**(-n) * trace((Theta @ W)**n).

The paper also decomposes each cycle moment into its cycle cumulant and a
reducible contribution assembled from *chain* cumulants.  Both equivalent
calculations are implemented here.

Matrix powers count weighted closed directed walks.  Nodes and edges need not
be distinct along the walk.

Terminology warning
-------------------
This module implements the one-index PRE closed-walk cycle family. It is not
the two-index PLOS path-pair trace family implemented in
:mod:`motif_cumulants.mixed_trace` / :mod:`motif_cumulants.covariance_motifs`,
and its order-2 cumulant is not generally identical to the binary SONET
reciprocal excess in :mod:`motif_cumulants.second_order`.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, TypedDict

import numpy as np

from ._validation import (
    is_sparse_matrix,
    prepare_adjacency,
    validate_max_order,
    validate_real_vector,
)
from .chain import _projector_cumulants_from_prepared_matrix


CycleCumulantMethod = Literal["auto", "projector", "moments"]


class CycleMotifResult(TypedDict, total=False):
    """Dictionary returned by :func:`cycle_motif_cumulants`."""

    order: np.ndarray
    cumulants: np.ndarray
    moments: np.ndarray
    chain_cumulants: np.ndarray


def _cycle_moments_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
) -> np.ndarray:
    """Compute cycle moments after adjacency validation."""
    scaled_matrix = matrix / n_nodes
    power = scaled_matrix.copy()
    moments = np.empty(max_order, dtype=np.float64)

    for index in range(max_order):
        if is_sparse_matrix(power):
            moments[index] = float(np.asarray(power.diagonal()).sum())
        else:
            moments[index] = float(np.trace(power))

        if index + 1 < max_order:
            power = power @ scaled_matrix

    return moments


def cycle_motif_moments(W: Any, max_order: int) -> np.ndarray:
    """Calculate cycle motif moments ``mu_cycle_1, ..., mu_cycle_K``.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix.  The package convention is
        ``W[i, j]`` = weight of ``j -> i``.  Since cycle statistics are based
        on traces, transposing an adjacency convention does not change them.
        Dense NumPy-compatible arrays and SciPy sparse matrices are accepted.
    max_order:
        Highest cycle order to compute.  Order is the number of directed edges
        in the closed walk.

    Returns
    -------
    numpy.ndarray
        One-dimensional array whose entry ``n - 1`` is
        ``trace(W**n) / N**n``.
    """
    max_order = validate_max_order(max_order)
    matrix, n_nodes = prepare_adjacency(W)
    return _cycle_moments_from_prepared_matrix(matrix, n_nodes, max_order)


def cycle_reducible_terms_from_chain_cumulants(
    chain_cumulants: Any,
) -> np.ndarray:
    """Calculate the chain-generated part of every cycle moment.

    If ``r_n`` denotes the reducible contribution at order ``n``, Eq. (20) of
    the paper gives

    ``r_n = sum_(n1,...,nt in C(n)) (n/t) * prod_i kappa_ni``.

    Directly enumerating ordered compositions is exponential.  The equivalent
    quadratic-time recurrence used here is

    ``r_n = n*kappa_n + sum_(j=1)^(n-1) kappa_j*r_(n-j)``.

    Parameters
    ----------
    chain_cumulants:
        One-dimensional sequence ``[kappa_1, ..., kappa_K]``.

    Returns
    -------
    numpy.ndarray
        Sequence ``[r_1, ..., r_K]``.
    """
    chain_array = validate_real_vector(
        chain_cumulants,
        name="chain_cumulants",
    )
    max_order = chain_array.size
    reducible = np.empty(max_order, dtype=np.float64)

    for index in range(max_order):
        order = index + 1
        value = order * chain_array[index]

        if index > 0:
            value += float(
                np.dot(
                    chain_array[:index],
                    reducible[index - 1 :: -1],
                )
            )

        reducible[index] = value

    return reducible


def cycle_cumulants_from_moments(
    cycle_moments: Any,
    chain_cumulants: Any,
) -> np.ndarray:
    """Convert cycle moments and chain cumulants into cycle cumulants.

    The paper defines

    ``mu_cycle_n = reducible_n + kappa_cycle_n``,

    where ``reducible_n`` is built from chain cumulants using Eq. (20).

    Parameters
    ----------
    cycle_moments:
        One-dimensional sequence ``[mu_cycle_1, ..., mu_cycle_K]``.
    chain_cumulants:
        One-dimensional sequence containing at least
        ``[kappa_chain_1, ..., kappa_chain_K]``.

    Returns
    -------
    numpy.ndarray
        Sequence ``[kappa_cycle_1, ..., kappa_cycle_K]``.
    """
    cycle_array = validate_real_vector(
        cycle_moments,
        name="cycle_moments",
    )
    chain_array = validate_real_vector(
        chain_cumulants,
        name="chain_cumulants",
    )

    if chain_array.size < cycle_array.size:
        raise ValueError(
            "chain_cumulants must contain at least as many orders as "
            "cycle_moments."
        )

    reducible = cycle_reducible_terms_from_chain_cumulants(
        chain_array[: cycle_array.size]
    )
    return cycle_array - reducible


def _projector_cycle_cumulants_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
) -> np.ndarray:
    """Compute cycle cumulants with ``trace((Theta W / N)**n)``."""
    if is_sparse_matrix(matrix):
        warnings.warn(
            "method='projector' converts a sparse adjacency matrix to dense. "
            "Use method='moments' or method='auto' to retain sparse matrix "
            "operations.",
            RuntimeWarning,
            stacklevel=3,
        )
        scaled_matrix = matrix.toarray() / n_nodes
    else:
        scaled_matrix = np.array(matrix / n_nodes, dtype=np.float64, copy=True)

    u = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)

    # Theta @ A = A - u @ (u.T @ A), with A = W / N.
    projected_matrix = scaled_matrix - np.outer(u, u @ scaled_matrix)
    power = projected_matrix.copy()
    cumulants = np.empty(max_order, dtype=np.float64)

    for index in range(max_order):
        cumulants[index] = float(np.trace(power))
        if index + 1 < max_order:
            power = power @ projected_matrix

    return cumulants


def cycle_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    method: CycleCumulantMethod = "auto",
    return_moments: bool = True,
    return_chain_cumulants: bool = False,
) -> CycleMotifResult:
    """Calculate cycle motif cumulants through a requested maximum order.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix.  Dense NumPy-compatible arrays and
        SciPy sparse matrices are accepted.
    max_order:
        Calculate ``kappa_cycle_1, ..., kappa_cycle_max_order``.
    method:
        ``"projector"`` directly evaluates
        ``trace((Theta @ W)**n) / N**n``.  Applying the projector generally
        makes a sparse matrix dense.

        ``"moments"`` computes the cycle moments and removes the contribution
        generated by chain cumulants using the paper's composition formula.

        ``"auto"`` (default) uses ``"projector"`` for dense matrices and
        ``"moments"`` for sparse matrices.
    return_moments:
        Include the cycle moments in the returned dictionary.
    return_chain_cumulants:
        Include the chain cumulants used in the cycle decomposition.

    Returns
    -------
    CycleMotifResult
        Dictionary with keys ``"order"`` and ``"cumulants"``.  Optional keys
        are ``"moments"`` and ``"chain_cumulants"``.

    Notes
    -----
    Cycle cumulants are defined for every order ``n >= 1``.  Thus,
    ``kappa_cycle_1 = mu_cycle_1 - kappa_chain_1``; it is not simply the
    self-loop frequency.
    """
    max_order = validate_max_order(max_order)
    if method not in ("auto", "projector", "moments"):
        raise ValueError("method must be 'auto', 'projector', or 'moments'.")
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")
    if not isinstance(return_chain_cumulants, (bool, np.bool_)):
        raise TypeError("return_chain_cumulants must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "moments" if is_sparse_matrix(matrix) else "projector"
        )

    cycle_moments: Optional[np.ndarray] = None
    chain_cumulants: Optional[np.ndarray] = None

    if selected_method == "projector":
        cycle_cumulants = _projector_cycle_cumulants_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
        )
    else:
        cycle_moments = _cycle_moments_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
        )
        chain_cumulants = _projector_cumulants_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
        )
        cycle_cumulants = cycle_cumulants_from_moments(
            cycle_moments,
            chain_cumulants,
        )

    result: CycleMotifResult = {
        "order": np.arange(1, max_order + 1, dtype=int),
        "cumulants": cycle_cumulants,
    }

    if return_moments:
        if cycle_moments is None:
            cycle_moments = _cycle_moments_from_prepared_matrix(
                matrix,
                n_nodes,
                max_order,
            )
        result["moments"] = cycle_moments

    if return_chain_cumulants:
        if chain_cumulants is None:
            chain_cumulants = _projector_cumulants_from_prepared_matrix(
                matrix,
                n_nodes,
                max_order,
            )
        result["chain_cumulants"] = chain_cumulants

    return result
