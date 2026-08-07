"""Null-network generators and directed-triad enrichment.

Classical network motifs are usually defined by over- or under-representation
relative to an explicit null ensemble. This module provides several topology
nulls while keeping them separate from the exact motif-cumulant identities.

Available nulls
---------------
``density_matched_null``
    Preserves the exact number of directed off-diagonal edges.

``directed_degree_preserving_null``
    Uses directed double-edge swaps to preserve every node's binary in- and
    out-degree exactly.

``block_density_matched_null``
    Preserves the exact edge count in every target-group/source-group block.

``shuffle_edge_weights``
    Preserves topology and shuffles the multiset of nonzero weights.

``triad_enrichment``
    Compares the 16-class induced triad census with one of the binary null
    ensembles and returns z-scores and empirical two-sided p-values.

References
----------
Milo et al. (2002), *Network motifs: Simple building blocks of complex
networks*, Science 298, 824-827.
https://doi.org/10.1126/science.298.5594.824

Zhao et al. (2011), *Synchronization from Second Order Network Connectivity
Statistics*, Frontiers in Computational Neuroscience 5:28.
https://doi.org/10.3389/fncom.2011.00028

Adjacency convention
--------------------
``W[i, j]`` is the edge ``j -> i``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, Optional, TypedDict
import warnings

import numpy as np

from ._validation import is_sparse_matrix, prepare_adjacency
from .triads import (
    EdgeRule,
    TRIAD_NAMES,
    _validate_edge_rule,
    _validate_threshold,
    directed_triad_census,
)


NullModelName = Literal["density", "degree", "block"]


class TriadEnrichmentResult(TypedDict, total=False):
    """Result returned by :func:`triad_enrichment`."""

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


def _rng(random_state: Any) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _dense_matrix(W: Any) -> tuple[np.ndarray, int]:
    matrix, n_nodes = prepare_adjacency(W)
    if is_sparse_matrix(matrix):
        return matrix.toarray(), n_nodes
    return np.array(matrix, dtype=np.float64, copy=True), n_nodes


def _topology(
    matrix: np.ndarray,
    *,
    edge_threshold: float,
    edge_rule: EdgeRule,
) -> np.ndarray:
    if edge_rule == "absolute":
        mask = np.abs(matrix) > edge_threshold
    else:
        mask = matrix > edge_threshold
    mask = np.asarray(mask, dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def _validate_preserve_weights(value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("preserve_weight_distribution must be Boolean.")
    return bool(value)


def _assign_sampled_edges(
    shape: tuple[int, int],
    selected_positions: np.ndarray,
    original_weights: np.ndarray,
    *,
    preserve_weight_distribution: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float64)
    if selected_positions.size == 0:
        return result
    if preserve_weight_distribution:
        weights = np.array(original_weights, dtype=np.float64, copy=True)
        rng.shuffle(weights)
    else:
        weights = np.ones(selected_positions.shape[0], dtype=np.float64)
    result[selected_positions[:, 0], selected_positions[:, 1]] = weights
    return result


def density_matched_null(
    W: Any,
    *,
    random_state: Any = None,
    edge_threshold: float = 0.0,
    edge_rule: EdgeRule = "absolute",
    preserve_weight_distribution: bool = False,
) -> np.ndarray:
    """Sample a loop-free graph with the same exact number of directed edges.

    By default the result is binary. With
    ``preserve_weight_distribution=True``, the original detected edge weights
    are randomly assigned to the sampled edge positions.
    """
    threshold = _validate_threshold(edge_threshold)
    edge_rule = _validate_edge_rule(edge_rule)
    preserve = _validate_preserve_weights(preserve_weight_distribution)
    rng = _rng(random_state)
    matrix, n_nodes = _dense_matrix(W)
    mask = _topology(matrix, edge_threshold=threshold, edge_rule=edge_rule)

    candidates = np.argwhere(~np.eye(n_nodes, dtype=bool))
    n_edges = int(mask.sum())
    chosen = (
        rng.choice(candidates.shape[0], size=n_edges, replace=False)
        if n_edges
        else np.empty(0, dtype=int)
    )
    positions = candidates[chosen]
    original_weights = matrix[mask]
    return _assign_sampled_edges(
        matrix.shape,
        positions,
        original_weights,
        preserve_weight_distribution=preserve,
        rng=rng,
    )


def _validate_swap_count(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def directed_degree_preserving_null(
    W: Any,
    *,
    n_swaps: Optional[int] = None,
    max_tries: Optional[int] = None,
    random_state: Any = None,
    edge_threshold: float = 0.0,
    edge_rule: EdgeRule = "absolute",
    preserve_weight_distribution: bool = False,
    strict: bool = False,
) -> np.ndarray:
    """Randomize topology while preserving binary in- and out-degrees.

    A valid directed double-edge swap replaces ``a -> b`` and ``c -> d`` by
    ``a -> d`` and ``c -> b`` when this creates neither self-loops nor duplicate
    edges. The operation preserves every source out-degree and target
    in-degree.

    ``preserve_weight_distribution`` preserves only the global multiset of
    edge weights, not weighted node strengths. The weights are shuffled after
    topology randomization.
    """
    threshold = _validate_threshold(edge_threshold)
    edge_rule = _validate_edge_rule(edge_rule)
    preserve = _validate_preserve_weights(preserve_weight_distribution)
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict must be Boolean.")
    rng = _rng(random_state)
    matrix, n_nodes = _dense_matrix(W)
    mask = _topology(matrix, edge_threshold=threshold, edge_rule=edge_rule)

    # Store conventional source -> target pairs. np.argwhere(mask) returns
    # package-oriented (target, source) coordinates.
    edges = [(int(source), int(target)) for target, source in np.argwhere(mask)]
    edge_set = set(edges)
    n_edges = len(edges)
    target_swaps = 10 * n_edges if n_swaps is None else _validate_swap_count(
        n_swaps, name="n_swaps"
    )
    tries_limit = (
        max(100, 100 * target_swaps)
        if max_tries is None
        else _validate_swap_count(max_tries, name="max_tries")
    )

    completed = 0
    attempts = 0
    while completed < target_swaps and attempts < tries_limit and n_edges >= 2:
        attempts += 1
        first, second = rng.choice(n_edges, size=2, replace=False)
        source_a, target_b = edges[int(first)]
        source_c, target_d = edges[int(second)]

        if source_a == source_c or target_b == target_d:
            continue
        new_first = (source_a, target_d)
        new_second = (source_c, target_b)
        if source_a == target_d or source_c == target_b:
            continue
        if new_first in edge_set or new_second in edge_set:
            continue

        edge_set.remove((source_a, target_b))
        edge_set.remove((source_c, target_d))
        edge_set.add(new_first)
        edge_set.add(new_second)
        edges[int(first)] = new_first
        edges[int(second)] = new_second
        completed += 1

    if completed < target_swaps:
        message = (
            f"Completed {completed} of {target_swaps} requested directed "
            f"edge swaps after {attempts} attempts. The degree sequence may "
            "admit few valid swaps."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    positions = np.array(
        [(target, source) for source, target in edges],
        dtype=int,
    ).reshape(-1, 2)
    return _assign_sampled_edges(
        (n_nodes, n_nodes),
        positions,
        matrix[mask],
        preserve_weight_distribution=preserve,
        rng=rng,
    )


def _group_indices(groups: Any, n_nodes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    raw = np.asarray(groups, dtype=object)
    if raw.ndim != 1 or raw.shape[0] != n_nodes:
        raise ValueError(f"groups must be one-dimensional with length {n_nodes}.")
    labels: list[Any] = []
    mapping: dict[Any, int] = {}
    inverse = np.empty(n_nodes, dtype=int)
    for node, label in enumerate(raw.tolist()):
        try:
            exists = label in mapping
        except TypeError as exc:
            raise TypeError("Every group label must be hashable.") from exc
        if not exists:
            mapping[label] = len(labels)
            labels.append(label)
        inverse[node] = mapping[label]
    nodes = [np.flatnonzero(inverse == index) for index in range(len(labels))]
    return np.asarray(labels, dtype=object), nodes


def block_density_matched_null(
    W: Any,
    groups: Any,
    *,
    random_state: Any = None,
    edge_threshold: float = 0.0,
    edge_rule: EdgeRule = "absolute",
    preserve_weight_distribution: bool = False,
) -> np.ndarray:
    """Preserve exact edge counts within every population block.

    A block is indexed by target population (row block) and source population
    (column block), following the package adjacency convention. Self-loops are
    excluded even within same-population blocks.
    """
    threshold = _validate_threshold(edge_threshold)
    edge_rule = _validate_edge_rule(edge_rule)
    preserve = _validate_preserve_weights(preserve_weight_distribution)
    rng = _rng(random_state)
    matrix, n_nodes = _dense_matrix(W)
    mask = _topology(matrix, edge_threshold=threshold, edge_rule=edge_rule)
    _, nodes_by_group = _group_indices(groups, n_nodes)
    result = np.zeros_like(matrix, dtype=np.float64)

    for target_nodes in nodes_by_group:
        for source_nodes in nodes_by_group:
            candidates = np.array(
                [
                    (target, source)
                    for target in target_nodes
                    for source in source_nodes
                    if target != source
                ],
                dtype=int,
            ).reshape(-1, 2)
            if candidates.shape[0] == 0:
                continue
            block_mask = mask[np.ix_(target_nodes, source_nodes)]
            if np.array_equal(target_nodes, source_nodes):
                block_mask = block_mask.copy()
                np.fill_diagonal(block_mask, False)
            n_edges = int(block_mask.sum())
            if n_edges == 0:
                continue
            selected = candidates[
                rng.choice(candidates.shape[0], size=n_edges, replace=False)
            ]
            if preserve:
                weights = matrix[
                    np.ix_(target_nodes, source_nodes)
                ][block_mask]
                weights = np.array(weights, copy=True)
                rng.shuffle(weights)
            else:
                weights = np.ones(n_edges, dtype=np.float64)
            result[selected[:, 0], selected[:, 1]] = weights
    return result


def shuffle_edge_weights(
    W: Any,
    *,
    random_state: Any = None,
) -> np.ndarray:
    """Shuffle nonzero edge weights while preserving their exact positions."""
    rng = _rng(random_state)
    matrix, _ = _dense_matrix(W)
    support = matrix != 0.0
    weights = np.array(matrix[support], copy=True)
    rng.shuffle(weights)
    result = np.zeros_like(matrix)
    result[support] = weights
    return result


def _validate_n_random(n_random: Any) -> int:
    if isinstance(n_random, (bool, np.bool_)) or not isinstance(
        n_random, Integral
    ):
        raise TypeError("n_random must be an integer of at least 2.")
    n_random = int(n_random)
    if n_random < 2:
        raise ValueError("n_random must be at least 2.")
    return n_random


def triad_enrichment(
    W: Any,
    *,
    n_random: int = 100,
    null_model: NullModelName = "degree",
    groups: Any = None,
    random_state: Any = None,
    edge_threshold: float = 0.0,
    edge_rule: EdgeRule = "absolute",
    n_swaps: Optional[int] = None,
    max_tries: Optional[int] = None,
    return_samples: bool = False,
    sample_size: Optional[int] = None,
) -> TriadEnrichmentResult:
    """Compare an induced triad census with a randomized null ensemble.

    The z-score is ``(observed - null_mean) / null_std``. It is ``NaN`` when
    the sampled null count has zero variance. Empirical p-values use a
    two-sided deviation from the sampled null mean and the standard ``+1``
    finite-sample correction.

    Parameters
    ----------
    W:
        Square adjacency matrix (``W[i, j]`` = edge j → i).
    n_random:
        Number of null-model networks to generate.  Must be ≥ 2.
    null_model:
        ``'density'`` preserves the exact edge count; ``'degree'`` preserves
        every node's binary in- and out-degree via directed double-edge swaps;
        ``'block'`` preserves edge counts within every group × group block
        (requires ``groups``).
    groups:
        Node group labels; required when ``null_model='block'``.
    random_state:
        Integer seed, ``numpy.random.Generator``, or ``None``.
    edge_threshold:
        Edges with ``|weight| ≤ edge_threshold`` are treated as absent.
    edge_rule:
        ``'absolute'`` uses ``|weight|``; ``'positive'`` uses ``weight``.
    n_swaps:
        Number of directed double-edge swaps for ``null_model='degree'``.
        Defaults to ``10 × n_edges``.
    max_tries:
        Maximum swap attempts for ``null_model='degree'``.
    return_samples:
        If ``True``, include the raw ``(n_random, 16)`` null-count matrix as
        ``null_samples`` in the result.
    sample_size:
        When given, :func:`directed_triad_census` uses Monte Carlo sampling of
        this many random triples instead of exact O(N³) enumeration for every
        census call (observed network and all ``n_random`` null networks).
        Using the same ``sample_size`` for all calls keeps the z-scores
        comparable.  ``None`` (default) uses exact enumeration.  A value
        around 200 000 is sufficient for stable z-scores in networks of
        N ~ 100–300 and avoids the multi-minute runtimes of the exact path.

    Returns
    -------
    TriadEnrichmentResult
    """
    n_random = _validate_n_random(n_random)
    if null_model not in ("density", "degree", "block"):
        raise ValueError("null_model must be 'density', 'degree', or 'block'.")
    if null_model == "block" and groups is None:
        raise ValueError("groups are required for null_model='block'.")
    if not isinstance(return_samples, (bool, np.bool_)):
        raise TypeError("return_samples must be Boolean.")
    if sample_size is not None and (
        not isinstance(sample_size, int) or sample_size < 1
    ):
        raise ValueError("sample_size must be a positive integer or None.")

    threshold = _validate_threshold(edge_threshold)
    edge_rule = _validate_edge_rule(edge_rule)
    rng = _rng(random_state)

    _census_kwargs = dict(
        edge_threshold=threshold,
        edge_rule=edge_rule,
        sample_size=sample_size,
        random_state=rng,
    )

    observed = directed_triad_census(W, **_census_kwargs)
    samples = np.empty((n_random, len(TRIAD_NAMES)), dtype=np.float64)

    for index in range(n_random):
        if null_model == "density":
            randomized = density_matched_null(
                W,
                random_state=rng,
                edge_threshold=threshold,
                edge_rule=edge_rule,
            )
        elif null_model == "degree":
            randomized = directed_degree_preserving_null(
                W,
                n_swaps=n_swaps,
                max_tries=max_tries,
                random_state=rng,
                edge_threshold=threshold,
                edge_rule=edge_rule,
            )
        else:
            randomized = block_density_matched_null(
                W,
                groups,
                random_state=rng,
                edge_threshold=threshold,
                edge_rule=edge_rule,
            )
        samples[index] = directed_triad_census(randomized, **_census_kwargs)["counts"]

    null_mean = samples.mean(axis=0)
    null_std = samples.std(axis=0, ddof=1)
    z_score = np.full(len(TRIAD_NAMES), np.nan, dtype=np.float64)
    variable = null_std > 0.0
    z_score[variable] = (
        observed["counts"][variable] - null_mean[variable]
    ) / null_std[variable]

    observed_deviation = np.abs(observed["counts"] - null_mean)
    sample_deviation = np.abs(samples - null_mean[None, :])
    p_value = (
        1.0 + np.sum(sample_deviation >= observed_deviation[None, :], axis=0)
    ) / (n_random + 1.0)

    result: TriadEnrichmentResult = {
        "triad": np.asarray(TRIAD_NAMES),
        "observed_counts": observed["counts"],
        "observed_fractions": observed["fractions"],
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "empirical_p_two_sided": p_value,
        "n_random": n_random,
        "null_model": null_model,
    }
    if return_samples:
        result["null_samples"] = samples
    return result

# ---------------------------------------------------------------------------
# Backward-compatible high-level null-model API.
# ---------------------------------------------------------------------------
NullModel = Literal["degree_preserving", "fixed_edges", "erdos_renyi"]


def _legacy_edge_rule(edge_presence: str) -> EdgeRule:
    if edge_presence == "nonzero":
        return "absolute"
    if edge_presence == "positive":
        return "positive"
    raise ValueError("edge_presence must be 'nonzero' or 'positive'.")


def randomize_directed_adjacency(
    W: Any,
    *,
    null_model: NullModel = "degree_preserving",
    n_swaps: Optional[int] = None,
    max_tries: Optional[int] = None,
    random_state: Any = None,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
) -> np.ndarray:
    """Generate a binary directed null graph using a named ensemble.

    Parameters
    ----------
    null_model:
        ``"degree_preserving"`` uses directed double-edge swaps;
        ``"fixed_edges"`` samples uniformly with the same exact edge count;
        ``"erdos_renyi"`` samples every off-diagonal edge independently at
        the observed density.

    Returns
    -------
    numpy.ndarray
        Dense Boolean adjacency matrix in package orientation
        ``result[target, source]``.
    """
    if null_model not in ("degree_preserving", "fixed_edges", "erdos_renyi"):
        raise ValueError(
            "null_model must be 'degree_preserving', 'fixed_edges', or "
            "'erdos_renyi'."
        )
    threshold = _validate_threshold(threshold)
    rule = _legacy_edge_rule(edge_presence)
    if n_swaps is not None:
        n_swaps = _validate_swap_count(n_swaps, name="n_swaps")
    if max_tries is not None:
        max_tries = _validate_swap_count(max_tries, name="max_tries")
    if n_swaps is not None and max_tries is not None and max_tries < n_swaps:
        raise ValueError("max_tries must be at least n_swaps.")

    if null_model == "degree_preserving":
        randomized = directed_degree_preserving_null(
            W,
            n_swaps=n_swaps,
            max_tries=max_tries,
            random_state=random_state,
            edge_threshold=threshold,
            edge_rule=rule,
        )
    elif null_model == "fixed_edges":
        randomized = density_matched_null(
            W,
            random_state=random_state,
            edge_threshold=threshold,
            edge_rule=rule,
        )
    else:
        rng = _rng(random_state)
        matrix, n_nodes = _dense_matrix(W)
        mask = _topology(
            matrix,
            edge_threshold=threshold,
            edge_rule=rule,
        )
        possible = n_nodes * (n_nodes - 1)
        probability = float(mask.sum() / possible) if possible else 0.0
        randomized = rng.random((n_nodes, n_nodes)) < probability
        np.fill_diagonal(randomized, False)
    return np.asarray(randomized, dtype=bool)


def triad_motif_enrichment(
    W: Any,
    *,
    n_random: int = 100,
    null_model: NullModel = "degree_preserving",
    n_swaps: Optional[int] = None,
    max_tries: Optional[int] = None,
    random_state: Any = None,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
    return_null_counts: bool = False,
    sample_size: Optional[int] = None,
) -> dict:
    """Calculate classical triad enrichment under a named null ensemble.

    In addition to z-scores, this compatibility interface reports the unit-
    norm *triad significance profile* introduced in classical motif analysis.
    A zero vector is returned when all finite z-scores are zero.

    Parameters
    ----------
    sample_size:
        Passed to :func:`directed_triad_census` for Monte Carlo sampling.
        ``None`` (default) uses exact O(N³) enumeration.  See
        :func:`triad_enrichment` for details.
    """
    n_random = _validate_n_random(n_random)
    if not isinstance(return_null_counts, (bool, np.bool_)):
        raise TypeError("return_null_counts must be Boolean.")
    if sample_size is not None and (
        not isinstance(sample_size, int) or sample_size < 1
    ):
        raise ValueError("sample_size must be a positive integer or None.")
    threshold = _validate_threshold(threshold)
    _legacy_edge_rule(edge_presence)
    rng = _rng(random_state)

    _census_kwargs = dict(
        threshold=threshold,
        edge_presence=edge_presence,
        sample_size=sample_size,
        random_state=rng,
    )

    observed = directed_triad_census(W, **_census_kwargs)
    null_counts = np.empty((n_random, len(TRIAD_NAMES)), dtype=np.float64)
    for index in range(n_random):
        randomized = randomize_directed_adjacency(
            W,
            null_model=null_model,
            n_swaps=n_swaps,
            max_tries=max_tries,
            random_state=rng,
            threshold=threshold,
            edge_presence=edge_presence,
        )
        null_counts[index] = directed_triad_census(randomized, **_census_kwargs)["counts"]

    null_mean = null_counts.mean(axis=0)
    null_std = null_counts.std(axis=0, ddof=1)
    z_scores = np.full(len(TRIAD_NAMES), np.nan, dtype=np.float64)
    variable = null_std > 0.0
    z_scores[variable] = (
        observed["counts"][variable] - null_mean[variable]
    ) / null_std[variable]

    centered_observed = np.abs(observed["counts"] - null_mean)
    centered_null = np.abs(null_counts - null_mean[None, :])
    empirical_p = (
        1.0 + np.sum(centered_null >= centered_observed[None, :], axis=0)
    ) / (n_random + 1.0)

    finite_z = np.nan_to_num(z_scores, nan=0.0)
    norm = float(np.linalg.norm(finite_z))
    significance_profile = finite_z / norm if norm > 0.0 else finite_z

    result: dict[str, Any] = {
        "triad": np.asarray(TRIAD_NAMES),
        "observed_counts": observed["counts"],
        "observed_proportions": observed["proportions"],
        "null_mean": null_mean,
        "null_std": null_std,
        "z_scores": z_scores,
        "z_score": z_scores,
        "significance_profile": significance_profile,
        "empirical_p_two_sided": empirical_p,
        "n_random": n_random,
        "null_model": null_model,
    }
    if return_null_counts:
        result["null_counts"] = null_counts
        result["null_samples"] = null_counts
    return result
