"""Shared validation helpers for motif-cumulant calculations."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

try:  # SciPy is optional and is needed only for sparse inputs.
    from scipy import sparse as scipy_sparse
except ImportError:  # pragma: no cover - depends on the user's environment.
    scipy_sparse = None


def validate_max_order(max_order: int) -> int:
    """Validate and normalize the requested maximum motif order."""
    if isinstance(max_order, (bool, np.bool_)) or not isinstance(
        max_order, Integral
    ):
        raise TypeError("max_order must be an integer.")

    max_order = int(max_order)
    if max_order < 1:
        raise ValueError("max_order must be at least 1.")
    return max_order


def is_sparse_matrix(value: Any) -> bool:
    """Return whether ``value`` is a SciPy sparse matrix."""
    return scipy_sparse is not None and scipy_sparse.issparse(value)


def prepare_adjacency(W: Any) -> tuple[Any, int]:
    """Return a finite, square, float64 adjacency matrix and its size."""
    if is_sparse_matrix(W):
        if len(W.shape) != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square two-dimensional matrix.")
        if W.shape[0] == 0:
            raise ValueError("W must contain at least one node.")
        if np.issubdtype(W.dtype, np.complexfloating):
            raise TypeError("Complex-valued adjacency matrices are not supported.")

        matrix = W.astype(np.float64, copy=False).tocsr()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("W contains NaN or infinite edge weights.")
        return matrix, int(matrix.shape[0])

    array = np.asarray(W)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("W must be a square two-dimensional matrix.")
    if array.shape[0] == 0:
        raise ValueError("W must contain at least one node.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("Complex-valued adjacency matrices are not supported.")

    try:
        matrix = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("W must contain numeric edge weights.") from exc

    if not np.all(np.isfinite(matrix)):
        raise ValueError("W contains NaN or infinite edge weights.")
    return matrix, int(matrix.shape[0])


def validate_real_vector(values: Any, *, name: str) -> np.ndarray:
    """Validate a nonempty one-dimensional finite real numeric sequence."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if raw.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"Complex-valued {name} are not supported.")

    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def prepare_real_vector(
    values: Any,
    *,
    n_nodes: int,
    name: str,
    normalize: bool = False,
) -> np.ndarray:
    """Validate a finite real vector of length ``n_nodes``.

    When ``normalize`` is true, the vector is divided by its Euclidean norm.
    A zero vector is never accepted because it cannot define an input,
    readout, or motif-weighting direction.
    """
    vector = validate_real_vector(values, name=name)
    if vector.size != n_nodes:
        raise ValueError(f"{name} must have shape ({n_nodes},).")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must not be the zero vector.")
    if normalize:
        vector = vector / norm
    return np.asarray(vector, dtype=np.float64)

