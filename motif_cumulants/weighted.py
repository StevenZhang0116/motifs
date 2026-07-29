"""User-friendly aliases for generalized input/readout chain cumulants.

The canonical implementation lives in :mod:`motif_cumulants.generalized` and
follows Hu et al. (2018), Supplementary Eqs. S41-S42:
https://doi.org/10.1103/PhysRevE.98.062312

This module provides the synonymous names ``input_output_*`` and
``weighted_*`` without duplicating the numerical implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .generalized import (
    GeneralizedChainMotifResult,
    generalized_chain_motif_cumulants,
    generalized_chain_motif_moments,
)


InputOutputChainResult = GeneralizedChainMotifResult


def input_output_chain_motif_moments(
    W: Any,
    max_order: int,
    *,
    B: Any,
    C: Any,
) -> np.ndarray:
    """Alias for :func:`generalized_chain_motif_moments`."""
    return generalized_chain_motif_moments(
        W,
        max_order,
        B=B,
        C=C,
    )


def input_output_chain_motif_cumulants(
    W: Any,
    max_order: int,
    *,
    B: Any,
    C: Any,
    return_moments: bool = True,
) -> GeneralizedChainMotifResult:
    """Alias for :func:`generalized_chain_motif_cumulants`."""
    return generalized_chain_motif_cumulants(
        W,
        max_order,
        B=B,
        C=C,
        return_moments=return_moments,
    )


weighted_chain_motif_moments = input_output_chain_motif_moments
weighted_chain_motif_cumulants = input_output_chain_motif_cumulants
