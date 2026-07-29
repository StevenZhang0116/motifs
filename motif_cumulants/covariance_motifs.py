"""Unified covariance-relevant motif calculations.

The covariance and dimensionality expansion of Recanatesi et al. (2019)
contains several motif families simultaneously: chain, divergent, convergent,
and mixed trace cumulants. This module provides a high-level function that
calculates them with one consistent maximum path order while preserving the
specialized functions in :mod:`chain`, :mod:`branching`, and
:mod:`mixed_trace`.

Reference
---------
Recanatesi, Ocker, Buice, and Shea-Brown (2019), "Dimensionality in recurrent
spiking networks: Global trends in activity and local origins in
connectivity", PLOS Computational Biology 15(7), e1006446.
https://doi.org/10.1371/journal.pcbi.1006446
Supplementary equations: https://doi.org/10.1371/journal.pcbi.1006446.s001

Important terminology
---------------------
The PLOS *mixed trace* quantity is indexed by two path lengths ``(n, m)`` and
contains ``Tr(W^n (W.T)^m)``. It is not the same as the PRE closed-walk cycle
cumulant ``Tr((Theta W)^n)`` in :mod:`motif_cumulants.cycle`.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import numpy as np

from .branching import (
    BranchingMotifResult,
    convergent_motif_cumulants,
    divergent_motif_cumulants,
)
from .chain import ChainMotifResult, chain_motif_cumulants
from .mixed_trace import (
    MixedTraceMotifResult,
    TraceNormalization,
    mixed_trace_motif_cumulants,
    mixed_trace_motif_moments,
)
from .second_order import SecondOrderMotifResult, second_order_motif_statistics
from ._validation import validate_max_order


class CovarianceMotifResult(TypedDict, total=False):
    """Result returned by :func:`covariance_motif_cumulants`.

    Nested results are available under ``chain``, ``divergent``,
    ``convergent``, and ``trace``. Flat array aliases are also returned for
    convenient numerical work and backward compatibility.
    """

    path_order: np.ndarray
    total_order: np.ndarray
    chain: ChainMotifResult
    divergent: BranchingMotifResult
    convergent: BranchingMotifResult
    trace: MixedTraceMotifResult
    second_order: SecondOrderMotifResult
    chain_cumulants: np.ndarray
    divergent_cumulants: np.ndarray
    convergent_cumulants: np.ndarray
    trace_cumulants: np.ndarray
    chain_moments: np.ndarray
    divergent_moments: np.ndarray
    convergent_moments: np.ndarray
    trace_moments: np.ndarray
    moments: dict[str, np.ndarray]
    cumulants: dict[str, np.ndarray]


def trace_motif_moments(
    W: Any,
    max_order: int,
    *,
    normalization: TraceNormalization = "recanatesi",
) -> np.ndarray:
    """Alias with explicit PLOS terminology for mixed trace moments.

    This wrapper is equivalent to :func:`mixed_trace_motif_moments`.
    """
    return mixed_trace_motif_moments(
        W, max_order, normalization=normalization
    )


def trace_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    normalization: TraceNormalization = "recanatesi",
    return_moments: bool = True,
) -> MixedTraceMotifResult:
    """Alias with explicit PLOS terminology for mixed trace cumulants.

    This wrapper is equivalent to :func:`mixed_trace_motif_cumulants` and is
    deliberately distinct from :func:`cycle_motif_cumulants`.
    """
    return mixed_trace_motif_cumulants(
        W,
        max_order,
        normalization=normalization,
        return_moments=return_moments,
    )


def covariance_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    trace_normalization: TraceNormalization = "recanatesi",
    return_moments: bool = True,
    include_trace: bool = True,
    include_second_order: bool = True,
) -> CovarianceMotifResult:
    """Calculate the principal global motif families in the PLOS expansion.

    Parameters
    ----------
    max_order:
        Maximum chain order and maximum length of each branch/path in the
        two-index divergent, convergent, and trace arrays.
    trace_normalization:
        Normalization passed to :func:`trace_motif_cumulants`.
    return_moments:
        Include raw moments in each nested result.
    include_trace:
        Mixed trace cumulants require a dense projector. Set false to avoid
        this potentially expensive calculation for a large sparse matrix.
    include_second_order:
        Include the compact weighted walk-based chain/divergent/convergent/
        reciprocal profile for convenient low-order inspection.

    Returns
    -------
    CovarianceMotifResult
        Nested dictionaries under ``chain``, ``divergent``, ``convergent``,
        and optionally ``trace`` and ``second_order``.
    """
    max_order = validate_max_order(max_order)
    for name, value in (
        ("return_moments", return_moments),
        ("include_trace", include_trace),
        ("include_second_order", include_second_order),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a Boolean value.")

    orders = np.arange(1, max_order + 1, dtype=int)
    chain = chain_motif_cumulants(
        W, max_order, return_moments=return_moments
    )
    divergent = divergent_motif_cumulants(
        W, max_order, return_moments=return_moments
    )
    convergent = convergent_motif_cumulants(
        W, max_order, return_moments=return_moments
    )

    result: CovarianceMotifResult = {
        "path_order": orders,
        "chain": chain,
        "divergent": divergent,
        "convergent": convergent,
    }

    trace: Optional[MixedTraceMotifResult] = None
    if include_trace:
        trace = trace_motif_cumulants(
            W,
            max_order,
            normalization=trace_normalization,
            return_moments=return_moments,
        )
        result["trace"] = trace

    # The compact second-order section is the expanded convenience interface.
    # Alongside it, expose flat aliases and grouped array dictionaries while
    # preserving the minimal nested result when all optional sections are off.
    if include_second_order:
        result["second_order"] = second_order_motif_statistics(W)
        result["total_order"] = np.add.outer(orders, orders)
        result["chain_cumulants"] = chain["cumulants"]
        result["divergent_cumulants"] = divergent["cumulants"]
        result["convergent_cumulants"] = convergent["cumulants"]
        result["cumulants"] = {
            "chain": chain["cumulants"],
            "divergent": divergent["cumulants"],
            "convergent": convergent["cumulants"],
        }
        result["moments"] = {}
        if return_moments:
            result["chain_moments"] = chain["moments"]
            result["divergent_moments"] = divergent["moments"]
            result["convergent_moments"] = convergent["moments"]
            result["moments"].update(
                {
                    "chain": chain["moments"],
                    "divergent": divergent["moments"],
                    "convergent": convergent["moments"],
                }
            )
        if trace is not None:
            result["trace_cumulants"] = trace["cumulants"]
            result["cumulants"]["trace"] = trace["cumulants"]
            if return_moments:
                result["trace_moments"] = trace["moments"]
                result["moments"]["trace"] = trace["moments"]
    return result
