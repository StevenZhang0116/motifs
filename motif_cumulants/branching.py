"""Convergent and divergent motif moments and cumulants.

These two-path motif families are used in covariance expansions in:

Recanatesi, Ocker, Buice, and Shea-Brown (2019), "Dimensionality in recurrent
spiking networks: Global trends in activity and local origins in
connectivity", PLOS Computational Biology 15(7), e1006446.
https://doi.org/10.1371/journal.pcbi.1006446
Supplementary equations: https://doi.org/10.1371/journal.pcbi.1006446.s001

They build on the motif-cumulant framework of Hu et al. (2014):
https://doi.org/10.1103/PhysRevE.89.032802

Adjacency convention
--------------------
``W[i, j]`` is the weight of the directed edge ``j -> i``.

Let ``A = W / N``, let ``u`` be a unit-norm node-weight vector (uniform by
default), and let ``Theta_u = I - u @ u.T``. For branch lengths
``n, m >= 1``:

Divergent motifs (two paths leaving a common source)

    mu_div[n,m] = u.T @ A**n @ (A.T)**m @ u

    kappa_div[n,m]
        = u.T @ B_n @ Theta_u @ B_m.T @ u,

Convergent motifs (two paths arriving at a common target)

    mu_conv[n,m] = u.T @ (A.T)**n @ A**m @ u

    kappa_conv[n,m]
        = u.T @ B_n.T @ Theta_u @ B_m @ u,

where ``B_n = (A @ Theta_u)**(n - 1) @ A``.

The returned arrays are indexed as ``[n - 1, m - 1]``. These are weighted
walk statistics; repeated nodes are allowed. They are not induced-subgraph
counts. Passing ``weights`` implements the nonuniform unit-vector weighting
discussed in the PLOS paper; the supplied vector is normalized internally.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np

from ._validation import (
    prepare_adjacency,
    prepare_real_vector,
    validate_max_order,
)


BranchingKind = Literal["divergent", "convergent"]


class BranchingMotifResult(TypedDict, total=False):
    """Dictionary returned by the branching motif cumulant functions."""

    branch_order: np.ndarray
    total_order: np.ndarray
    cumulants: np.ndarray
    moments: np.ndarray
    weight_vector: np.ndarray


def _unit_weight_vector(weights: Any, n_nodes: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)
    return prepare_real_vector(
        weights,
        n_nodes=n_nodes,
        name="weights",
        normalize=True,
    )


def _branch_vectors(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    *,
    kind: BranchingKind,
    projected: bool,
    unit_vector: np.ndarray,
) -> np.ndarray:
    """Return one branch vector per order after adjacency validation.

    For convergent motifs, row ``n - 1`` is ``A**n @ u`` for moments and
    ``B_n @ u`` for cumulants. For divergent motifs, it is the corresponding
    transposed propagation, ``(A.T)**n @ u`` or ``B_n.T @ u``.
    """
    scaled = matrix / n_nodes
    operator = scaled if kind == "convergent" else scaled.T

    vectors = np.empty((max_order, n_nodes), dtype=np.float64)
    propagated = np.asarray(
        operator @ unit_vector, dtype=np.float64
    ).reshape(-1)

    for index in range(max_order):
        vectors[index] = propagated
        if index + 1 < max_order:
            if projected:
                propagated = propagated - unit_vector * float(
                    np.dot(unit_vector, propagated)
                )
            propagated = np.asarray(
                operator @ propagated,
                dtype=np.float64,
            ).reshape(-1)

    return vectors


def _gram_with_optional_projection(
    vectors: np.ndarray,
    *,
    project_between_branches: bool,
    unit_vector: np.ndarray,
) -> np.ndarray:
    """Form the branch-vector Gram matrix, optionally inserting Theta."""
    if not project_between_branches:
        return vectors @ vectors.T

    overlaps = vectors @ unit_vector
    return vectors @ vectors.T - np.outer(overlaps, overlaps)


def _branching_moments_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    *,
    kind: BranchingKind,
    unit_vector: np.ndarray,
) -> np.ndarray:
    vectors = _branch_vectors(
        matrix,
        n_nodes,
        max_order,
        kind=kind,
        projected=False,
        unit_vector=unit_vector,
    )
    return _gram_with_optional_projection(
        vectors,
        project_between_branches=False,
        unit_vector=unit_vector,
    )


def _branching_cumulants_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
    *,
    kind: BranchingKind,
    unit_vector: np.ndarray,
) -> np.ndarray:
    vectors = _branch_vectors(
        matrix,
        n_nodes,
        max_order,
        kind=kind,
        projected=True,
        unit_vector=unit_vector,
    )
    return _gram_with_optional_projection(
        vectors,
        project_between_branches=True,
        unit_vector=unit_vector,
    )


def divergent_motif_moments(
    W: Any,
    max_order: int,
    *,
    weights: Any = None,
) -> np.ndarray:
    """Calculate moments of two paths leaving a common source.

    Entry ``[n - 1, m - 1]`` is
    ``u.T @ W**n @ (W.T)**m @ u / N**(n + m)``, where ``u`` is uniform
    unless ``weights`` is supplied.
    """
    max_order = validate_max_order(max_order)
    matrix, n_nodes = prepare_adjacency(W)
    unit_vector = _unit_weight_vector(weights, n_nodes)
    return _branching_moments_from_prepared_matrix(
        matrix,
        n_nodes,
        max_order,
        kind="divergent",
        unit_vector=unit_vector,
    )


def convergent_motif_moments(
    W: Any,
    max_order: int,
    *,
    weights: Any = None,
) -> np.ndarray:
    """Calculate moments of two paths arriving at a common target.

    Entry ``[n - 1, m - 1]`` is
    ``u.T @ (W.T)**n @ W**m @ u / N**(n + m)``, where ``u`` is uniform
    unless ``weights`` is supplied.
    """
    max_order = validate_max_order(max_order)
    matrix, n_nodes = prepare_adjacency(W)
    unit_vector = _unit_weight_vector(weights, n_nodes)
    return _branching_moments_from_prepared_matrix(
        matrix,
        n_nodes,
        max_order,
        kind="convergent",
        unit_vector=unit_vector,
    )


def _branching_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    kind: BranchingKind,
    return_moments: bool,
    weights: Any,
) -> BranchingMotifResult:
    max_order = validate_max_order(max_order)
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    unit_vector = _unit_weight_vector(weights, n_nodes)
    cumulants = _branching_cumulants_from_prepared_matrix(
        matrix,
        n_nodes,
        max_order,
        kind=kind,
        unit_vector=unit_vector,
    )

    orders = np.arange(1, max_order + 1, dtype=int)
    result: BranchingMotifResult = {
        "branch_order": orders,
        "total_order": np.add.outer(orders, orders),
        "cumulants": cumulants,
        "weight_vector": unit_vector,
    }

    if return_moments:
        result["moments"] = _branching_moments_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
            kind=kind,
            unit_vector=unit_vector,
        )

    return result


def divergent_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    return_moments: bool = True,
    weights: Any = None,
) -> BranchingMotifResult:
    """Calculate divergent motif cumulants for all branch-length pairs.

    ``result["cumulants"][n - 1, m - 1]`` is the irreducible statistic for
    two paths of lengths ``n`` and ``m`` leaving a common source.
    """
    return _branching_motif_cumulants(
        W,
        max_order,
        kind="divergent",
        return_moments=return_moments,
        weights=weights,
    )


def convergent_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    return_moments: bool = True,
    weights: Any = None,
) -> BranchingMotifResult:
    """Calculate convergent motif cumulants for all branch-length pairs.

    ``result["cumulants"][n - 1, m - 1]`` is the irreducible statistic for
    two paths of lengths ``n`` and ``m`` arriving at a common target.
    """
    return _branching_motif_cumulants(
        W,
        max_order,
        kind="convergent",
        return_moments=return_moments,
        weights=weights,
    )
