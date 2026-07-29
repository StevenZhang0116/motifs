"""Second-order directed motif profiles in two complementary conventions.

This module deliberately separates two calculations that are often both
called "second-order motifs":

``second_order_motif_statistics``
    Weighted, ``N``-normalized matrix/walk statistics.  Repeated node indices
    are allowed, matching the normalization used by the path-cumulant modules.

``sonet_motif_statistics``
    Exact loop-free binary counts on distinct node positions, following the
    second-order network (SONET) convention of Zhao et al.  It reports chain,
    divergent, convergent, and reciprocal motif frequencies relative to an
    independent-edge baseline.

References
----------
Zhao, Beverlin, Netoff, and Nykamp (2011), "Synchronization from Second Order
Network Connectivity Statistics", Frontiers in Computational Neuroscience.
https://doi.org/10.3389/fncom.2011.00028

Hu et al. (2018), "Feedback through graph motifs relates structure and
function in complex networks", Physical Review E 98, 062312.
https://doi.org/10.1103/PhysRevE.98.062312

The SONET reciprocal excess below is not generally identical to the order-2
closed-walk *cycle cumulant* of Hu et al.  The latter removes a different set
of lower-order chain contributions and is returned separately by the weighted
profile for comparison.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from ._binary import EdgePresence, prepare_binary_adjacency
from ._validation import is_sparse_matrix, prepare_adjacency


SECOND_ORDER_MOTIF_NAMES = np.array(
    ["chain", "divergent", "convergent", "reciprocal"]
)


class SecondOrderMotifResult(TypedDict):
    """Dictionary returned by :func:`second_order_motif_statistics`."""

    connection: float
    erdos_renyi_baseline: float
    motif: np.ndarray
    moments: np.ndarray
    cumulants: np.ndarray
    independent_edge_excess: np.ndarray
    cycle_order_2_cumulant: float
    self_loops_removed: bool


class SONETMotifResult(TypedDict):
    """Dictionary returned by :func:`sonet_motif_statistics`."""

    n_nodes: int
    n_edges: int
    connection_probability: float
    motif: np.ndarray
    counts: np.ndarray
    possible: np.ndarray
    frequencies: np.ndarray
    independent_edge_baseline: np.ndarray
    independent_edge_excess: np.ndarray
    alpha: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    threshold: float
    edge_presence: str
    self_loops_removed: bool


def _remove_diagonal(matrix: Any) -> Any:
    if is_sparse_matrix(matrix):
        copied = matrix.copy().tocsr()
        copied.setdiag(0.0)
        copied.eliminate_zeros()
        return copied

    copied = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(copied, 0.0)
    return copied


def _trace_of_square(matrix: Any) -> float:
    square = matrix @ matrix
    if is_sparse_matrix(square):
        return float(np.asarray(square.diagonal()).sum())
    return float(np.trace(square))


def second_order_motif_statistics(
    W: Any,
    *,
    remove_self_loops: bool = False,
) -> SecondOrderMotifResult:
    """Calculate weighted chain, divergent, convergent, and reciprocal terms.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix with ``W[i, j]`` representing
        ``j -> i``.
    remove_self_loops:
        If true, calculate all statistics after setting the diagonal to zero.
        The input matrix itself is never modified.

    Returns
    -------
    SecondOrderMotifResult
        ``moments`` and ``cumulants`` follow the order in ``motif``:
        ``chain``, ``divergent``, ``convergent``, and ``reciprocal``.

        Here ``cumulants = moments - p**2`` is the independent-edge excess
        used in compact SONET-style summaries.  The identical array is also
        available under the more explicit key ``independent_edge_excess``.
        ``cycle_order_2_cumulant`` separately reports
        ``Tr((Theta W / N)**2)``, matching :func:`cycle_motif_cumulants`.

    Notes
    -----
    The matrix formulas average over all ordered node indices, so repeated
    nodes are allowed.  For binary loop-free networks this differs from an
    exact distinct-node subgraph census by finite-size corrections.  Use
    :func:`sonet_motif_statistics` for exact distinct-node frequencies.
    """
    if not isinstance(remove_self_loops, (bool, np.bool_)):
        raise TypeError("remove_self_loops must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    if remove_self_loops:
        matrix = _remove_diagonal(matrix)

    scaled = matrix / n_nodes
    u = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)

    incoming = np.asarray(scaled @ u, dtype=np.float64).reshape(-1)
    outgoing = np.asarray(scaled.T @ u, dtype=np.float64).reshape(-1)

    connection = float(np.dot(u, incoming))
    chain = float(np.dot(u, scaled @ incoming))
    divergent = float(np.dot(outgoing, outgoing))
    convergent = float(np.dot(incoming, incoming))
    reciprocal = _trace_of_square(scaled)

    moments = np.array(
        [chain, divergent, convergent, reciprocal],
        dtype=np.float64,
    )
    baseline = connection**2
    excess = moments - baseline

    # Match cycle_motif_cumulants(W, max_order=2, method="projector") at
    # order 2, without importing the public cycle module.
    if is_sparse_matrix(scaled):
        scaled_dense = scaled.toarray()
    else:
        scaled_dense = np.asarray(scaled)
    theta_scaled = scaled_dense - np.outer(u, u @ scaled_dense)
    cycle_order_2 = float(np.trace(theta_scaled @ theta_scaled))

    return {
        "connection": connection,
        "erdos_renyi_baseline": baseline,
        "motif": SECOND_ORDER_MOTIF_NAMES.copy(),
        "moments": moments,
        "cumulants": excess,
        "independent_edge_excess": excess.copy(),
        "cycle_order_2_cumulant": cycle_order_2,
        "self_loops_removed": bool(remove_self_loops),
    }


def sonet_motif_statistics(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: EdgePresence = "nonzero",
) -> SONETMotifResult:
    """Count exact distinct-node SONET motifs in a directed binary graph.

    The weighted input is first converted to a loop-free binary adjacency
    matrix.  With the package convention ``W[target, source]``, the exact
    counts are

    ``chain``
        ``sum_i in_degree[i] * out_degree[i] - 2 * reciprocal_count``;
        the subtraction removes two-node backtracking paths.
    ``divergent``
        ``sum_i choose(out_degree[i], 2)``.
    ``convergent``
        ``sum_i choose(in_degree[i], 2)``.
    ``reciprocal``
        Number of unordered node pairs connected in both directions.

    Each count is divided by the exact number of possible distinct-node
    placements.  If ``p`` is the directed edge density, the independent-edge
    baseline is ``p**2`` and the SONET coefficient is

    ``alpha = frequency / p**2 - 1``.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix with ``W[target, source]`` for
        ``source -> target``.
    threshold:
        Strict threshold used to declare an edge.
    edge_presence:
        ``"nonzero"`` uses ``abs(W) > threshold``; ``"positive"`` uses
        ``W > threshold``.  Self-loops are always ignored.

    Returns
    -------
    SONETMotifResult
        Exact counts, possible placements, frequencies, independent-edge
        excesses, and SONET ``alpha`` values in the order
        ``chain``, ``divergent``, ``convergent``, ``reciprocal``.

    Notes
    -----
    A frequency or ``alpha`` is ``NaN`` when its denominator is zero, for
    example three-node motifs in a graph with fewer than three nodes or
    ``alpha`` in an empty graph.
    """
    binary, n_nodes = prepare_binary_adjacency(
        W,
        threshold=threshold,
        edge_presence=edge_presence,
        remove_self_loops=True,
    )

    in_degree = binary.sum(axis=1, dtype=np.int64)
    out_degree = binary.sum(axis=0, dtype=np.int64)
    n_edges = int(binary.sum())

    mutual_edge_matrix = binary & binary.T
    reciprocal_count = int(mutual_edge_matrix.sum() // 2)
    divergent_count = int(np.sum(out_degree * (out_degree - 1) // 2))
    convergent_count = int(np.sum(in_degree * (in_degree - 1) // 2))
    chain_count = int(
        np.dot(in_degree.astype(np.int64), out_degree.astype(np.int64))
        - 2 * reciprocal_count
    )

    counts = np.array(
        [
            chain_count,
            divergent_count,
            convergent_count,
            reciprocal_count,
        ],
        dtype=np.int64,
    )

    possible_directed_edges = n_nodes * (n_nodes - 1)
    possible_three_node_ordered = n_nodes * (n_nodes - 1) * (n_nodes - 2)
    possible = np.array(
        [
            possible_three_node_ordered,
            possible_three_node_ordered // 2,
            possible_three_node_ordered // 2,
            n_nodes * (n_nodes - 1) // 2,
        ],
        dtype=np.int64,
    )

    connection_probability = (
        n_edges / possible_directed_edges
        if possible_directed_edges > 0
        else float("nan")
    )

    frequencies = np.full(4, np.nan, dtype=np.float64)
    np.divide(
        counts,
        possible,
        out=frequencies,
        where=possible > 0,
    )

    baseline_value = connection_probability**2
    baseline = np.full(4, baseline_value, dtype=np.float64)
    excess = frequencies - baseline

    alpha = np.full(4, np.nan, dtype=np.float64)
    if np.isfinite(baseline_value) and baseline_value > 0.0:
        alpha = frequencies / baseline_value - 1.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "connection_probability": float(connection_probability),
        "motif": SECOND_ORDER_MOTIF_NAMES.copy(),
        "counts": counts,
        "possible": possible,
        "frequencies": frequencies,
        "independent_edge_baseline": baseline,
        "independent_edge_excess": excess,
        "alpha": alpha,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "threshold": float(threshold),
        "edge_presence": str(edge_presence),
        "self_loops_removed": True,
    }
