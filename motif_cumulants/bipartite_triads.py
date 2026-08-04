"""Directed-triad enrichment for one-way rectangular (bipartite) networks.

A rectangular binary matrix records connections from one population of source
nodes to a separate population of target nodes.  Because the two populations
are disjoint and no within-population edges exist, this is a directed
bipartite network rather than a general directed graph.

The quantity of interest is an observed-to-random occurrence ratio

    R_M = observed_count[M] / mean_random_count[M],

not the probabilistic motif likelihood used by
:mod:`motif_cumulants.probabilistic_triads`.  Use this module when the matrix
entries are realized binary observations; use ``probabilistic_triads`` when
they are calibrated connection probabilities.

Structurally possible motifs
----------------------------
For a strictly one-way network the only nonempty induced triads are

``012``
    one source-to-target edge;
``021D``
    one source projecting to two targets (a divergent triplet);
``021U``
    two sources projecting to one target (a convergent triplet).

Every other directed triad class requires a reverse or within-population edge
and therefore cannot occur.  Those classes are reported as ``NaN`` ratios.

Method
------
Each rectangular network is lifted into a square block adjacency matrix whose
diagonal blocks are empty, and is compared with the block-constrained null of
:func:`motif_cumulants.null_models.block_density_matched_null`.  The block
constraint is essential: an unconstrained null would fabricate impossible
source-to-source and target-to-target edges.

Two rectangular matrices are analyzed separately and their relative
occurrences displayed side by side.  Note that this compares each network with
its own random baseline; it is not a significance test of one network against
the other, which would require replicates or a node/block bootstrap.

Adjacency convention
--------------------
``forward[target, source] == 1`` represents the edge ``source -> target``,
matching the package-wide ``W[i, j] = j -> i`` orientation.
"""

from __future__ import annotations

from math import comb
from typing import Any, TypedDict

import numpy as np

from .null_models import triad_enrichment


# Induced triad classes that a strictly one-way bipartite network can realize.
ONE_WAY_BIPARTITE_TRIAD_NAMES = ("003", "012", "021D", "021U")


class OneWayBipartiteRatioResult(TypedDict):
    """Result returned by :func:`one_way_bipartite_triplet_ratios`."""

    n_source: int
    n_target: int
    n_edges: int
    edge_probability: float
    observed_divergent: int
    expected_divergent: float
    divergent_ratio: float
    observed_convergent: int
    expected_convergent: float
    convergent_ratio: float


class BipartiteTriadEnrichmentResult(TypedDict, total=False):
    """Result returned by :func:`bipartite_triad_enrichment`."""

    triad: np.ndarray
    observed_counts: np.ndarray
    observed_fractions: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray
    z_score: np.ndarray
    empirical_p_two_sided: np.ndarray
    n_random: int
    null_model: str
    null_samples: np.ndarray
    mixed_observed_counts: np.ndarray
    mixed_null_mean: np.ndarray
    mixed_observed_fractions: np.ndarray
    mixed_null_fractions: np.ndarray
    mixed_null_samples: np.ndarray
    relative_occurrence: np.ndarray
    log2_relative_occurrence: np.ndarray
    structurally_possible: np.ndarray
    n_source: int
    n_target: int
    n_edges: int
    edge_density: float
    n_mixed_triplets: int
    structurally_empty_triplets: int


def _validate_binary_rectangular(W: Any, *, name: str) -> np.ndarray:
    """Validate and return a dense binary rectangular matrix."""
    matrix = np.asarray(W)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise TypeError(f"Complex-valued {name} is not supported.")
    try:
        numeric = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric entries.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or infinite entries.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only binary 0/1 entries.")
    return numeric.astype(np.uint8, copy=False)


def lift_bipartite_adjacency(forward: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lift a one-way rectangular network into a square adjacency matrix.

    Parameters
    ----------
    forward:
        Binary matrix of shape ``(n_target, n_source)`` where
        ``forward[target, source] == 1`` represents ``source -> target``.

    Returns
    -------
    full_adjacency:
        Square ``(n_source + n_target, n_source + n_target)`` matrix with nodes
        ordered as ``[source nodes, target nodes]`` and block structure

        .. code-block:: text

            [ 0        0 ]
            [ forward  0 ]

        following the package convention ``W[target, source]``.
    groups:
        Integer population labels (``0`` for sources, ``1`` for targets)
        suitable for ``triad_enrichment(..., null_model="block", groups=...)``.
    """
    forward_array = _validate_binary_rectangular(forward, name="forward")
    n_target, n_source = forward_array.shape

    full = np.zeros(
        (n_source + n_target, n_source + n_target),
        dtype=np.uint8,
    )
    # Source -> target occupies the lower-left block.
    full[n_source:, :n_source] = forward_array

    groups = np.concatenate(
        (
            np.zeros(n_source, dtype=int),
            np.ones(n_target, dtype=int),
        )
    )
    return full, groups


def one_way_bipartite_triplet_ratios(
    forward: Any,
) -> OneWayBipartiteRatioResult:
    """Analytical divergent/convergent enrichment under a Bernoulli edge null.

    For a one-way network the two nonempty wedge counts are determined exactly
    by the source out-degrees and target in-degrees:

    .. code-block:: text

        N_divergent  = sum_j choose(out_degree[j], 2)
        N_convergent = sum_i choose(in_degree[i], 2)

    Under an independent Bernoulli null with edge probability ``p`` equal to the
    observed density, their expectations are

    .. code-block:: text

        E[N_divergent]  = n_source * choose(n_target, 2) * p**2
        E[N_convergent] = n_target * choose(n_source, 2) * p**2

    This is far cheaper than :func:`bipartite_triad_enrichment` because it needs
    no randomization, but it provides no null variance, z-score, or p-value.
    Note also that the wedge counts are fixed by the degree sequence, so this
    Bernoulli comparison is the one that asks whether degree heterogeneity
    produces excess wedges; a degree-preserving null could not.

    Parameters
    ----------
    forward:
        Binary matrix of shape ``(n_target, n_source)``; see
        :func:`lift_bipartite_adjacency`.
    """
    forward_array = _validate_binary_rectangular(forward, name="forward")
    n_target, n_source = forward_array.shape
    counts = forward_array.astype(np.int64, copy=False)

    out_degree = counts.sum(axis=0)
    in_degree = counts.sum(axis=1)
    observed_divergent = int(np.sum(out_degree * (out_degree - 1) // 2))
    observed_convergent = int(np.sum(in_degree * (in_degree - 1) // 2))

    edge_probability = float(counts.mean())
    expected_divergent = n_source * comb(n_target, 2) * edge_probability**2
    expected_convergent = n_target * comb(n_source, 2) * edge_probability**2

    return {
        "n_source": n_source,
        "n_target": n_target,
        "n_edges": int(counts.sum()),
        "edge_probability": edge_probability,
        "observed_divergent": observed_divergent,
        "expected_divergent": expected_divergent,
        "divergent_ratio": (
            observed_divergent / expected_divergent
            if expected_divergent > 0.0
            else float("nan")
        ),
        "observed_convergent": observed_convergent,
        "expected_convergent": expected_convergent,
        "convergent_ratio": (
            observed_convergent / expected_convergent
            if expected_convergent > 0.0
            else float("nan")
        ),
    }


def bipartite_triad_enrichment(
    forward: Any,
    *,
    n_random: int = 100,
    random_state: Any = None,
    return_samples: bool = False,
) -> BipartiteTriadEnrichmentResult:
    """Compare one-way bipartite triplet motifs with block-randomized networks.

    Every randomized network preserves

    * the source and target population sizes;
    * the absence of within-population edges;
    * the exact number of source-to-target edges.

    Parameters
    ----------
    forward:
        Binary matrix of shape ``(n_target, n_source)``; see
        :func:`lift_bipartite_adjacency`.
    n_random:
        Number of randomized networks, at least 2.  The smallest attainable
        two-sided empirical p-value is ``1 / (n_random + 1)``.
    random_state:
        Seed or ``numpy.random.Generator``.
    return_samples:
        Also return the per-randomization triad counts.

    Returns
    -------
    BipartiteTriadEnrichmentResult
        ``relative_occurrence`` is ``observed / mean_random`` per triad class;
        values above one indicate enrichment.  Classes that the network cannot
        realize are ``NaN``; ``structurally_possible`` marks the finite ones.
        ``z_score`` and ``empirical_p_two_sided`` come from
        :func:`motif_cumulants.null_models.triad_enrichment`.

    Notes
    -----
    All-source and all-target triples are necessarily empty, so they are
    removed from the ``003`` count to give the ``mixed_*`` fields, which refer
    only to triples spanning both populations.  Because that correction is a
    deterministic constant shared by the observed census and every
    randomization, it leaves ``z_score``, ``null_std``, and the empirical
    p-values unchanged.

    The underlying triad census is ``O((n_source + n_target)**3)`` per
    randomization, so runtime grows quickly with ``n_random`` and population
    size.  For a one-way network the divergent and convergent counts are fully
    determined by the row and column degrees, so a degree-preserving null would
    fix them exactly; the block density-matched null used here is what makes
    the comparison informative.
    """
    forward_array = _validate_binary_rectangular(forward, name="forward")
    n_target, n_source = forward_array.shape

    n_mixed_triplets = (
        comb(n_source, 2) * n_target + n_source * comb(n_target, 2)
    )
    if n_mixed_triplets == 0:
        raise ValueError(
            "At least three nodes distributed across the two populations are "
            "required to define mixed bipartite triplets."
        )

    full, groups = lift_bipartite_adjacency(forward_array)
    base = triad_enrichment(
        full,
        n_random=n_random,
        null_model="block",
        groups=groups,
        random_state=random_state,
        return_samples=return_samples,
    )

    triad_names = np.asarray(base["triad"])
    empty_index = int(np.flatnonzero(triad_names == "003")[0])

    # Triples drawn entirely from one population are empty by construction.
    structurally_empty_triplets = comb(n_source, 3) + comb(n_target, 3)

    observed_mixed = np.asarray(base["observed_counts"], dtype=np.float64).copy()
    null_mixed_mean = np.asarray(base["null_mean"], dtype=np.float64).copy()
    observed_mixed[empty_index] -= structurally_empty_triplets
    null_mixed_mean[empty_index] -= structurally_empty_triplets

    relative_occurrence = np.divide(
        observed_mixed,
        null_mixed_mean,
        out=np.full_like(observed_mixed, np.nan),
        where=null_mixed_mean > 0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_relative_occurrence = np.log2(relative_occurrence)

    result: BipartiteTriadEnrichmentResult = dict(base)  # type: ignore[assignment]
    result.update(
        {
            "mixed_observed_counts": observed_mixed,
            "mixed_null_mean": null_mixed_mean,
            "mixed_observed_fractions": observed_mixed / n_mixed_triplets,
            "mixed_null_fractions": null_mixed_mean / n_mixed_triplets,
            "relative_occurrence": relative_occurrence,
            "log2_relative_occurrence": log2_relative_occurrence,
            "structurally_possible": np.isfinite(relative_occurrence),
            "n_source": n_source,
            "n_target": n_target,
            "n_edges": int(forward_array.sum()),
            "edge_density": float(forward_array.mean()),
            "n_mixed_triplets": n_mixed_triplets,
            "structurally_empty_triplets": structurally_empty_triplets,
        }
    )

    if return_samples:
        mixed_samples = np.asarray(base["null_samples"], dtype=np.float64).copy()
        mixed_samples[:, empty_index] -= structurally_empty_triplets
        result["mixed_null_samples"] = mixed_samples

    return result
