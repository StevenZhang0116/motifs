"""Input/readout-weighted chain motif moments and cumulants.

Hu et al. show that the chain resummation extends from uniform input/readout
vectors to arbitrary deterministic vectors ``B`` and ``C`` whenever
``C.T @ B`` is nonzero. Define

``N_BC = C.T @ B`` and ``Theta_BC = I - B C.T / N_BC``.

Then

``mu_n_BC = C.T W**n B / (N**n N_BC)``

and

``kappa_n_BC = C.T W (Theta_BC W)**(n-1) B / (N**n N_BC)``.

Reference
---------
Hu et al. (2018), *Feedback through graph motifs relates structure and
function in complex networks*, Physical Review E 98, 062312, Supplementary
Eqs. S41-S42.
https://doi.org/10.1103/PhysRevE.98.062312

Adjacency convention
--------------------
``W[i, j]`` is the weight of ``j -> i``. ``B`` weights where the signal enters;
``C`` weights how node activity is read out.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from ._validation import (
    prepare_adjacency,
    validate_max_order,
    validate_real_vector,
)


class GeneralizedChainMotifResult(TypedDict, total=False):
    """Result from :func:`generalized_chain_motif_cumulants`."""

    order: np.ndarray
    overlap: float
    cumulants: np.ndarray
    moments: np.ndarray


def _prepare_input_readout(
    B: Any,
    C: Any,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    B_array = validate_real_vector(B, name="B")
    C_array = validate_real_vector(C, name="C")
    if B_array.shape != (n_nodes,):
        raise ValueError(f"B must have shape ({n_nodes},).")
    if C_array.shape != (n_nodes,):
        raise ValueError(f"C must have shape ({n_nodes},).")

    overlap = float(np.dot(C_array, B_array))
    scale = float(np.linalg.norm(B_array) * np.linalg.norm(C_array))
    tolerance = 100.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    if abs(overlap) <= tolerance:
        raise ValueError(
            "C.T @ B must be nonzero for generalized motif normalization."
        )
    return B_array, C_array, overlap


def _generalized_moments_from_prepared(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    B: np.ndarray,
    C: np.ndarray,
    overlap: float,
) -> np.ndarray:
    scaled = matrix / n_nodes
    moments = np.empty(max_order, dtype=np.float64)
    propagated = B.copy()
    for index in range(max_order):
        propagated = np.asarray(scaled @ propagated).reshape(-1)
        moments[index] = float(np.dot(C, propagated) / overlap)
    return moments


def generalized_chain_motif_moments(
    W: Any,
    max_order: int,
    *,
    B: Any,
    C: Any,
) -> np.ndarray:
    """Calculate chain motif moments selected by input ``B`` and readout ``C``.

    Entry ``n-1`` equals ``C.T W**n B / (N**n * C.T B)``.
    Scaling either ``B`` or ``C`` by a nonzero constant does not change the
    normalized moments.
    """
    max_order = validate_max_order(max_order)
    matrix, n_nodes = prepare_adjacency(W)
    B_array, C_array, overlap = _prepare_input_readout(B, C, n_nodes)
    return _generalized_moments_from_prepared(
        matrix,
        n_nodes,
        max_order,
        B_array,
        C_array,
        overlap,
    )


def generalized_chain_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    B: Any,
    C: Any,
    return_moments: bool = True,
) -> GeneralizedChainMotifResult:
    """Calculate arbitrary-input/readout chain motif cumulants.

    The oblique projector ``Theta_BC`` is applied as a matrix-vector operation,
    so the implementation does not construct a dense projector and supports
    SciPy sparse adjacency matrices.

    When ``B = C = ones(N)/sqrt(N)``, the result is exactly the ordinary chain
    motif cumulant sequence returned by :func:`chain_motif_cumulants`.
    """
    max_order = validate_max_order(max_order)
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    B_array, C_array, overlap = _prepare_input_readout(B, C, n_nodes)
    scaled = matrix / n_nodes

    cumulants = np.empty(max_order, dtype=np.float64)
    propagated = np.asarray(scaled @ B_array).reshape(-1)

    for index in range(max_order):
        value = float(np.dot(C_array, propagated) / overlap)
        cumulants[index] = value
        if index + 1 < max_order:
            # Theta_BC @ propagated = propagated
            #     - B * (C.T @ propagated) / (C.T @ B).
            projected = propagated - B_array * value
            propagated = np.asarray(scaled @ projected).reshape(-1)

    result: GeneralizedChainMotifResult = {
        "order": np.arange(1, max_order + 1, dtype=int),
        "overlap": overlap,
        "cumulants": cumulants,
    }
    if return_moments:
        result["moments"] = _generalized_moments_from_prepared(
            matrix,
            n_nodes,
            max_order,
            B_array,
            C_array,
            overlap,
        )
    return result
