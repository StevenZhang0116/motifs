"""
Spectral statistics for directed adjacency matrices.

This module provides a concise spectral profile of an adjacency matrix: its
singular values, spectral radius, nuclear norm, and Frobenius norm.  Singular
values are the natural real-valued summary of a non-symmetric directed graph;
they equal the square roots of the eigenvalues of W^T W and capture both the
symmetric and antisymmetric components of the connectivity.

These statistics complement the walk-based cumulants and subgraph counts
provided by the rest of the package.  They are deliberately kept separate
because singular values summarise the global linear structure of the network,
whereas motif cumulants and triad counts characterise local connectivity
patterns.

Adjacency convention
--------------------
``W[i, j]`` represents ``j -> i``.  Edge weights are used as-is for the SVD;
no thresholding is applied.  To analyse a binary topology, threshold before
calling this function.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import numpy as np

from ._validation import is_sparse_matrix, prepare_adjacency


class SpectralStatisticsResult(TypedDict):
    """Dictionary returned by :func:`spectral_statistics`."""

    n_nodes: int
    singular_values: np.ndarray
    spectral_radius: float
    nuclear_norm: float
    frobenius_norm: float
    max_returned: Optional[int]
    self_loops_removed: bool


def spectral_statistics(
    W: Any,
    *,
    max_svd: Optional[int] = None,
    remove_self_loops: bool = False,
) -> SpectralStatisticsResult:
    """Compute the singular value spectrum of a directed adjacency matrix.

    Parameters
    ----------
    W:
        Square adjacency matrix with ``W[i, j]`` = weight of edge ``j → i``.
        Dense NumPy arrays and SciPy sparse matrices are both accepted.
    max_svd:
        If given, only the first ``max_svd`` singular values (largest first)
        are stored in ``singular_values``.  The scalar summaries
        ``spectral_radius``, ``nuclear_norm``, and ``frobenius_norm`` are
        always computed from the full spectrum.
    remove_self_loops:
        If ``True``, set the diagonal to zero before computing the SVD.
        The input matrix is never modified.

    Returns
    -------
    SpectralStatisticsResult
        ``singular_values`` are in descending order; when ``max_svd`` is set
        only the top ``max_svd`` values are stored.  ``spectral_radius``,
        ``nuclear_norm``, and ``frobenius_norm`` always reflect the full
        spectrum.

    Notes
    -----
    The singular values are the square roots of the eigenvalues of ``W^T W``.
    They are always real and non-negative:

    * ``spectral_radius`` = ``σ₁`` = operator (spectral) 2-norm ‖W‖₂.
    * ``nuclear_norm`` = ``Σ σᵢ`` = ‖W‖₊.
    * ``frobenius_norm`` = ``√(Σ σᵢ²)`` = ‖W‖_F.

    Full SVD is always computed internally (O(N³) for an N × N dense matrix).
    For large matrices this may be slow; ``max_svd`` only reduces output size,
    not runtime.
    """
    if not isinstance(remove_self_loops, (bool, np.bool_)):
        raise TypeError("remove_self_loops must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)

    if is_sparse_matrix(matrix):
        W_dense = np.asarray(matrix.toarray(), dtype=np.float64)
    else:
        W_dense = np.asarray(matrix, dtype=np.float64)

    if remove_self_loops:
        W_dense = W_dense.copy()
        np.fill_diagonal(W_dense, 0.0)

    svd = np.linalg.svd(W_dense, compute_uv=False)   # descending order

    spectral_radius = float(svd[0]) if svd.size > 0 else 0.0
    nuclear_norm    = float(svd.sum())
    frobenius_norm  = float(np.sqrt((svd ** 2).sum()))

    svd_out = svd[:max_svd] if max_svd is not None else svd

    return {
        "n_nodes":          n_nodes,
        "singular_values":  svd_out,
        "spectral_radius":  spectral_radius,
        "nuclear_norm":     nuclear_norm,
        "frobenius_norm":   frobenius_norm,
        "max_returned":     max_svd,
        "self_loops_removed": bool(remove_self_loops),
    }
