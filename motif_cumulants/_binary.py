"""Internal helpers for classical binary motif calculations.

The path-cumulant modules accept weighted matrices directly.  Classical
subgraph motifs, SONET frequencies, and null-model enrichment instead require
an edge-presence rule.  This module centralizes that conversion so every
binary calculation uses the same adjacency convention and thresholding.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Literal

import numpy as np

from ._validation import is_sparse_matrix, prepare_adjacency


EdgePresence = Literal["nonzero", "positive"]


def validate_edge_presence(edge_presence: str) -> EdgePresence:
    """Validate the rule used to convert weighted entries into edges."""
    if edge_presence not in ("nonzero", "positive"):
        raise ValueError("edge_presence must be 'nonzero' or 'positive'.")
    return edge_presence  # type: ignore[return-value]


def validate_threshold(threshold: float) -> float:
    """Validate a finite, nonnegative edge threshold."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, Real
    ):
        raise TypeError("threshold must be a real number.")

    value = float(threshold)
    if not np.isfinite(value):
        raise ValueError("threshold must be finite.")
    if value < 0.0:
        raise ValueError("threshold must be nonnegative.")
    return value


def prepare_binary_adjacency(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: EdgePresence = "nonzero",
    remove_self_loops: bool = True,
) -> tuple[np.ndarray, int]:
    """Return a dense Boolean adjacency matrix and its number of nodes.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix.  The package convention is
        ``W[target, source]`` for the edge ``source -> target``.
    threshold:
        Minimum edge magnitude or positive weight, depending on
        ``edge_presence``.  The comparison is strict.
    edge_presence:
        ``"nonzero"`` declares an edge when ``abs(W[i, j]) > threshold``.
        This is appropriate for signed networks when both signs represent
        anatomical or effective connections.  ``"positive"`` declares an
        edge only when ``W[i, j] > threshold``.
    remove_self_loops:
        Whether to set the diagonal to false after thresholding.  Classical
        directed triads and SONET motif counts conventionally exclude loops.

    Notes
    -----
    A dense Boolean matrix is returned intentionally.  Exact induced-triad
    enumeration is cubic in the number of nodes and requires fast random
    access to every dyad.  The weighted walk-cumulant modules retain sparse
    operations where feasible.
    """
    threshold = validate_threshold(threshold)
    edge_presence = validate_edge_presence(edge_presence)
    if not isinstance(remove_self_loops, (bool, np.bool_)):
        raise TypeError("remove_self_loops must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    dense = matrix.toarray() if is_sparse_matrix(matrix) else np.asarray(matrix)

    if edge_presence == "nonzero":
        binary = np.abs(dense) > threshold
    else:
        binary = dense > threshold

    binary = np.asarray(binary, dtype=bool)
    if remove_self_loops:
        binary = binary.copy()
        np.fill_diagonal(binary, False)

    return binary, n_nodes
