"""Chain motif moments and cumulants for directed weighted networks.

The implementation follows the motif-cumulant definitions used in
Hu et al., Physical Review E 98, 062312 (2018):
https://doi.org/10.1103/PhysRevE.98.062312

Adjacency convention
--------------------
``W[i, j]`` is the weight of the directed edge ``j -> i``.

For an ``N x N`` adjacency matrix ``W``, the order-``n`` chain motif moment is

    mu_n = 1 / N**(n + 1) * sum_{i,j} (W**n)[i, j].

The corresponding chain motif cumulant can be evaluated directly as

    kappa_n = 1 / N**n * u.T @ W @ (Theta @ W)**(n - 1) @ u,

where ``u = ones(N) / sqrt(N)`` and ``Theta = I - u @ u.T``.

The code never explicitly constructs ``Theta`` and does not form matrix powers.
Both dense NumPy arrays and SciPy sparse matrices are supported.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

import numpy as np

from ._validation import (
    prepare_adjacency,
    validate_max_order,
    validate_real_vector,
)


CumulantMethod = Literal["projector", "moments"]


class ChainMotifResult(TypedDict, total=False):
    """Dictionary returned by :func:`chain_motif_cumulants`."""

    order: np.ndarray
    cumulants: np.ndarray
    moments: np.ndarray


def _moments_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
) -> np.ndarray:
    """Compute moments after input validation has already been performed."""
    scaled_matrix = matrix / n_nodes
    u = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)

    moments = np.empty(max_order, dtype=np.float64)
    propagated = u.copy()

    for index in range(max_order):
        propagated = scaled_matrix @ propagated
        moments[index] = float(np.dot(u, propagated))

    return moments


def chain_motif_moments(W: Any, max_order: int) -> np.ndarray:
    """Calculate chain motif moments ``mu_1, ..., mu_max_order``.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix. ``W[i, j]`` denotes the edge
        ``j -> i``. Dense NumPy-compatible arrays and SciPy sparse matrices are
        accepted.
    max_order:
        Highest motif order to compute. Order is the number of directed edges
        in the chain.

    Returns
    -------
    numpy.ndarray
        One-dimensional array whose entry ``n - 1`` is ``mu_n``.
    """
    max_order = validate_max_order(max_order)
    matrix, n_nodes = prepare_adjacency(W)
    return _moments_from_prepared_matrix(matrix, n_nodes, max_order)


def chain_cumulants_from_moments(moments: Any) -> np.ndarray:
    """Convert chain motif moments into chain motif cumulants.

    This uses the ordered-composition recurrence

    ``kappa_n = mu_n - sum_{j=1}^{n-1} kappa_j * mu_{n-j}``.

    Parameters
    ----------
    moments:
        One-dimensional finite numeric sequence ``[mu_1, ..., mu_K]``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array ``[kappa_1, ..., kappa_K]``.
    """
    moment_array = validate_real_vector(moments, name="moments")

    cumulants = np.empty_like(moment_array)
    for n_index in range(moment_array.size):
        if n_index == 0:
            cumulants[n_index] = moment_array[n_index]
            continue

        reducible_part = np.dot(
            cumulants[:n_index],
            moment_array[n_index - 1 :: -1],
        )
        cumulants[n_index] = moment_array[n_index] - reducible_part

    return cumulants


def _projector_cumulants_from_prepared_matrix(
    matrix: Any,
    n_nodes: int,
    max_order: int,
) -> np.ndarray:
    """Compute chain cumulants with the direct projector formula."""
    scaled_matrix = matrix / n_nodes
    u = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)

    cumulants = np.empty(max_order, dtype=np.float64)

    # At order n, propagated equals
    #     (W/N) @ [Theta @ (W/N)]**(n - 1) @ u.
    propagated = scaled_matrix @ u

    for index in range(max_order):
        cumulant = float(np.dot(u, propagated))
        cumulants[index] = cumulant

        if index + 1 < max_order:
            # Apply Theta = I - u u^T without constructing a dense projector.
            projected = propagated - u * cumulant
            propagated = scaled_matrix @ projected

    return cumulants


def chain_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    method: CumulantMethod = "projector",
    return_moments: bool = True,
) -> ChainMotifResult:
    """Calculate chain motif cumulants through a requested maximum order.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix. ``W[i, j]`` denotes the edge
        ``j -> i``. Dense NumPy-compatible arrays and SciPy sparse matrices are
        accepted.
    max_order:
        Calculate ``kappa_1, ..., kappa_max_order``.
    method:
        ``"projector"`` evaluates the direct projector expression and is the
        recommended default. ``"moments"`` first computes all motif moments and
        then applies the ordered-composition recurrence.
    return_moments:
        Include ``mu_1, ..., mu_max_order`` in the returned dictionary.

    Returns
    -------
    ChainMotifResult
        Dictionary with keys ``"order"`` and ``"cumulants"``. It also contains
        ``"moments"`` when ``return_moments=True``.

    Examples
    --------
    >>> import numpy as np
    >>> W = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    >>> result = chain_motif_cumulants(W, max_order=4)
    >>> np.round(result["cumulants"], 12)
    array([0.33333333, 0.        , 0.        , 0.        ])
    """
    max_order = validate_max_order(max_order)
    if method not in ("projector", "moments"):
        raise ValueError("method must be either 'projector' or 'moments'.")
    if not isinstance(return_moments, (bool, np.bool_)):
        raise TypeError("return_moments must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    moments: Optional[np.ndarray] = None

    if method == "projector":
        cumulants = _projector_cumulants_from_prepared_matrix(
            matrix,
            n_nodes,
            max_order,
        )
    else:
        moments = _moments_from_prepared_matrix(matrix, n_nodes, max_order)
        cumulants = chain_cumulants_from_moments(moments)

    result: ChainMotifResult = {
        "order": np.arange(1, max_order + 1, dtype=int),
        "cumulants": cumulants,
    }

    if return_moments:
        if moments is None:
            moments = _moments_from_prepared_matrix(
                matrix,
                n_nodes,
                max_order,
            )
        result["moments"] = moments

    return result
