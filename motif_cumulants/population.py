"""Population-resolved chain and branching motif cumulants.

When nodes belong to known populations (for example excitatory/inhibitory cell
types or anatomical modules), scalar motif statistics can conceal where paths
start and end. Hu et al. extend motif cumulants using a block projector that
removes the uniform mode *within each population*.

This module implements direct block-averaged versions of that construction.
For population indicator matrix ``H`` and normalized indicator matrix ``U``,

``Theta_B = I - U U.T``.

Let ``B_n = (W Theta_B)**(n-1) W``. The population chain cumulant from source
population ``q`` to target population ``p`` is the block average of ``B_n``
divided by ``N**(n-1)``. Divergent and convergent branch matrices are formed
by inserting ``Theta_B`` between two projected path matrices.

Reference
---------
Hu, Trousdale, Josic, and Shea-Brown (2014), *Local paths to global coherence:
Cutting networks down to size*, Physical Review E 89, 032802.
https://doi.org/10.1103/PhysRevE.89.032802

Adjacency convention
--------------------
``W[i, j]`` is the weight of ``j -> i``. Therefore a chain result indexed
``[..., p, q]`` runs from source group ``q`` to target group ``p``.
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


PopulationBranchingKind = Literal["divergent", "convergent"]


class PopulationChainMotifResult(TypedDict, total=False):
    """Population-resolved chain motif result."""

    group: np.ndarray
    group_size: np.ndarray
    order: np.ndarray
    cumulants: np.ndarray
    moments: np.ndarray


class PopulationBranchingMotifResult(TypedDict, total=False):
    """Population-resolved divergent or convergent motif result."""

    group: np.ndarray
    group_size: np.ndarray
    branch_order: np.ndarray
    total_order: np.ndarray
    kind: str
    cumulants: np.ndarray
    moments: np.ndarray


class PopulationMotifResult(TypedDict):
    """Combined population chain/divergent/convergent result."""

    group: np.ndarray
    group_size: np.ndarray
    path_order: np.ndarray
    total_order: np.ndarray
    moments: dict[str, np.ndarray]
    cumulants: dict[str, np.ndarray]


def _prepare_groups(
    groups: Any,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(groups, dtype=object)
    if raw.ndim != 1 or raw.shape[0] != n_nodes:
        raise ValueError(f"groups must be one-dimensional with length {n_nodes}.")

    labels: list[Any] = []
    index_by_label: dict[Any, int] = {}
    inverse = np.empty(n_nodes, dtype=int)
    for node, label in enumerate(raw.tolist()):
        try:
            group_index = index_by_label.get(label)
        except TypeError as exc:
            raise TypeError("Every population label must be hashable.") from exc
        if group_index is None:
            group_index = len(labels)
            try:
                index_by_label[label] = group_index
            except TypeError as exc:
                raise TypeError("Every population label must be hashable.") from exc
            labels.append(label)
        inverse[node] = group_index

    n_groups = len(labels)
    indicator = np.zeros((n_nodes, n_groups), dtype=np.float64)
    indicator[np.arange(n_nodes), inverse] = 1.0
    sizes = indicator.sum(axis=0).astype(int)
    normalized = indicator / np.sqrt(sizes)[None, :]
    return np.asarray(labels, dtype=object), sizes, indicator, normalized


def _dense_adjacency(W: Any) -> tuple[np.ndarray, int]:
    matrix, n_nodes = prepare_adjacency(W)
    if is_sparse_matrix(matrix):
        warnings.warn(
            "Population cumulants use a dense block projector and therefore "
            "convert a sparse adjacency matrix to dense.",
            RuntimeWarning,
            stacklevel=3,
        )
        return matrix.toarray(), n_nodes
    return np.array(matrix, dtype=np.float64, copy=True), n_nodes


def _block_average(
    matrix: np.ndarray,
    indicator: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    totals = indicator.T @ matrix @ indicator
    return totals / np.outer(sizes, sizes)


def _path_bases(
    matrix: np.ndarray,
    theta_block: np.ndarray,
    max_order: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    powers: list[np.ndarray] = []
    projected: list[np.ndarray] = []
    power = matrix.copy()
    basis = matrix.copy()
    for index in range(max_order):
        powers.append(power)
        projected.append(basis)
        if index + 1 < max_order:
            power = power @ matrix
            basis = matrix @ theta_block @ basis
    return powers, projected


def _population_chain_arrays(
    powers: list[np.ndarray],
    projected: list[np.ndarray],
    indicator: np.ndarray,
    sizes: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_order = len(powers)
    n_groups = sizes.size
    moments = np.empty((max_order, n_groups, n_groups), dtype=np.float64)
    cumulants = np.empty_like(moments)
    for index, (power, basis) in enumerate(zip(powers, projected)):
        order = index + 1
        denominator = float(n_nodes ** (order - 1))
        moments[index] = _block_average(power, indicator, sizes) / denominator
        cumulants[index] = (
            _block_average(basis, indicator, sizes) / denominator
        )
    return moments, cumulants


def _population_branching_arrays(
    powers: list[np.ndarray],
    projected: list[np.ndarray],
    theta_block: np.ndarray,
    indicator: np.ndarray,
    sizes: np.ndarray,
    n_nodes: int,
    *,
    kind: PopulationBranchingKind,
) -> tuple[np.ndarray, np.ndarray]:
    max_order = len(powers)
    n_groups = sizes.size
    shape = (max_order, max_order, n_groups, n_groups)
    moments = np.empty(shape, dtype=np.float64)
    cumulants = np.empty(shape, dtype=np.float64)

    for n_index in range(max_order):
        n_order = n_index + 1
        for m_index in range(max_order):
            m_order = m_index + 1
            denominator = float(n_nodes ** (n_order + m_order - 1))
            if kind == "divergent":
                moment_matrix = powers[n_index] @ powers[m_index].T
                cumulant_matrix = (
                    projected[n_index]
                    @ theta_block
                    @ projected[m_index].T
                )
            else:
                moment_matrix = powers[n_index].T @ powers[m_index]
                cumulant_matrix = (
                    projected[n_index].T
                    @ theta_block
                    @ projected[m_index]
                )
            moments[n_index, m_index] = (
                _block_average(moment_matrix, indicator, sizes) / denominator
            )
            cumulants[n_index, m_index] = (
                _block_average(cumulant_matrix, indicator, sizes)
                / denominator
            )
    return moments, cumulants


def population_chain_motif_cumulants(
    W: Any,
    groups: Any,
    max_order: int,
    *,
    return_moments: bool = True,
) -> PopulationChainMotifResult:
    """Calculate source/target-population chain motif cumulants.

    ``result["cumulants"][n-1, p, q]`` is the order-``n`` chain cumulant
    from source group ``q`` to target group ``p``.
    """
    max_order = validate_max_order(max_order)
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")
    matrix, n_nodes = _dense_adjacency(W)
    labels, sizes, indicator, normalized = _prepare_groups(groups, n_nodes)
    theta_block = np.eye(n_nodes) - normalized @ normalized.T
    powers, projected = _path_bases(matrix, theta_block, max_order)
    moments, cumulants = _population_chain_arrays(
        powers, projected, indicator, sizes, n_nodes
    )

    result: PopulationChainMotifResult = {
        "group": labels,
        "group_size": sizes,
        "order": np.arange(1, max_order + 1, dtype=int),
        "cumulants": cumulants,
    }
    if return_moments:
        result["moments"] = moments
    return result


def population_branching_motif_cumulants(
    W: Any,
    groups: Any,
    max_order: int,
    *,
    kind: PopulationBranchingKind,
    return_moments: bool = True,
) -> PopulationBranchingMotifResult:
    """Calculate population-resolved divergent or convergent cumulants.

    The last two axes identify the two branch endpoints. For divergent motifs,
    the hidden common node is a source; for convergent motifs it is a target.
    """
    max_order = validate_max_order(max_order)
    if kind not in ("divergent", "convergent"):
        raise ValueError("kind must be 'divergent' or 'convergent'.")
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = _dense_adjacency(W)
    labels, sizes, indicator, normalized = _prepare_groups(groups, n_nodes)
    theta_block = np.eye(n_nodes) - normalized @ normalized.T
    powers, projected = _path_bases(matrix, theta_block, max_order)
    moments, cumulants = _population_branching_arrays(
        powers,
        projected,
        theta_block,
        indicator,
        sizes,
        n_nodes,
        kind=kind,
    )
    orders = np.arange(1, max_order + 1, dtype=int)
    result: PopulationBranchingMotifResult = {
        "group": labels,
        "group_size": sizes,
        "branch_order": orders,
        "total_order": np.add.outer(orders, orders),
        "kind": kind,
        "cumulants": cumulants,
    }
    if return_moments:
        result["moments"] = moments
    return result


def population_motif_cumulants(
    W: Any,
    groups: Any,
    max_order: int,
    *,
    return_moments: bool = True,
) -> PopulationMotifResult:
    """Calculate population chain, divergent, and convergent cumulants.

    This combined entry point reuses path powers and projected path bases, so
    it is more efficient than calling the three specialized functions
    separately.
    """
    max_order = validate_max_order(max_order)
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = _dense_adjacency(W)
    labels, sizes, indicator, normalized = _prepare_groups(groups, n_nodes)
    theta_block = np.eye(n_nodes) - normalized @ normalized.T
    powers, projected = _path_bases(matrix, theta_block, max_order)
    chain_moments, chain_cumulants = _population_chain_arrays(
        powers, projected, indicator, sizes, n_nodes
    )
    div_moments, div_cumulants = _population_branching_arrays(
        powers,
        projected,
        theta_block,
        indicator,
        sizes,
        n_nodes,
        kind="divergent",
    )
    conv_moments, conv_cumulants = _population_branching_arrays(
        powers,
        projected,
        theta_block,
        indicator,
        sizes,
        n_nodes,
        kind="convergent",
    )

    moments: dict[str, np.ndarray] = {}
    if return_moments:
        moments = {
            "chain": chain_moments,
            "divergent": div_moments,
            "convergent": conv_moments,
        }
    orders = np.arange(1, max_order + 1, dtype=int)
    return {
        "group": labels,
        "group_size": sizes,
        "path_order": orders,
        "total_order": np.add.outer(orders, orders),
        "moments": moments,
        "cumulants": {
            "chain": chain_cumulants,
            "divergent": div_cumulants,
            "convergent": conv_cumulants,
        },
    }
