"""Expected directed-triplet motif probabilities and random-network ratios.

This module implements the probability-based triplet analysis used by
Udvary et al. (2022), rather than a census of one realized binary network.
For a sampled triplet, the six possible directed edges are treated as
independent Bernoulli variables with probabilities supplied by ``P``.  The
probability of an induced labeled edge pattern is therefore

    product_e p_e**x_e * (1 - p_e)**(1 - x_e),

and the probability of a directed-triad isomorphism class is the sum over
all labeled patterns in that class.

Two random-network comparisons are returned:

1. Independent-edge baseline (Udvary et al., Fig. 3E and STAR Methods).
   Each of the six edge probabilities is replaced by its mean across sampled
   triplets before motif probabilities are recomputed.
2. Independent-dyad baseline (Song et al., 2005; Udvary et al., Fig. 6E).
   The mean frequencies of absent, one-way, and reciprocal doublets are
   preserved, but the three doublets constituting a triplet are combined
   independently.  Dividing by this baseline removes motif enrichment that
   is already explained by doublet statistics.

References
----------
Udvary et al. (2022), "The impact of neuron morphology on cortical network
architecture", Cell Reports 39, 110677.
https://doi.org/10.1016/j.celrep.2022.110677

Song et al. (2005), "Highly nonrandom features of synaptic connectivity in
local cortical circuits", PLOS Biology 3(3), e68.
https://doi.org/10.1371/journal.pbio.0030068

Reference implementation released with Udvary et al.:
https://github.com/zibneuro/udvary-et-al-2022/blob/master/structural_model/eval_motifs.py

Figure 6E doublet-normalization script released with Udvary et al.:
https://github.com/zibneuro/udvary-et-al-2022/blob/master/analysis/visualization/visualize_L5PTTripletMotifs.m

Adjacency convention
--------------------
``P[target, source]`` is the probability of the directed edge
``source -> target``.  Entries must lie in ``[0, 1]``.  Diagonal entries are
ignored because induced directed triads do not include self-loops.
"""

from __future__ import annotations

from itertools import combinations, islice
from math import comb
from numbers import Integral
from typing import Any, Iterator, Literal, Optional, TypedDict

import numpy as np

from ._validation import is_sparse_matrix, prepare_adjacency
from .triads import (
    TRIAD_NAMES,
    _EDGE_POSITIONS,
    _classify_outgoing_triad,
)


# Visual motif order used in Udvary et al. Figures 3E and 6E.  The empty
# triad 003 is omitted in the paper, leaving 15 nonempty classes.
UDVARY_TRIPLET_MOTIF_NAMES = (
    "300",
    "210",
    "120U",
    "120D",
    "120C",
    "030T",
    "030C",
    "201",
    "111U",
    "111D",
    "021U",
    "021D",
    "021C",
    "102",
    "012",
)

TRIPLET_EDGE_NAMES = (
    "0->1",
    "1->0",
    "0->2",
    "2->0",
    "1->2",
    "2->1",
)
TRIPLET_DYAD_NAMES = ("0<->1", "0<->2", "1<->2")
TRIPLET_DYAD_STATE_NAMES = (
    "absent",
    "forward_only",
    "reverse_only",
    "reciprocal",
)

DoubletBaseline = Literal["pooled", "position_specific"]


class TripletMotifProbabilityResult(TypedDict):
    """Result returned by :func:`triplet_motif_probability_ratios`."""

    motif_id: np.ndarray
    triad: np.ndarray
    model_probability: np.ndarray
    model_probability_standard_error: np.ndarray
    independent_edge_probability: np.ndarray
    relative_to_independent_edges: np.ndarray
    log2_relative_to_independent_edges: np.ndarray
    independent_dyad_probability: np.ndarray
    relative_to_independent_dyads: np.ndarray
    doublet_normalized_ratio: np.ndarray
    log2_relative_to_independent_dyads: np.ndarray
    mean_edge_probability: np.ndarray
    edge_position: np.ndarray
    mean_dyad_state_probability: np.ndarray
    independent_edge_dyad_state_probability: np.ndarray
    doublet_relative_occurrence: np.ndarray
    pooled_dyad_state_probability: np.ndarray
    pooled_independent_edge_dyad_state_probability: np.ndarray
    pooled_doublet_relative_occurrence: np.ndarray
    dyad_position: np.ndarray
    dyad_state: np.ndarray
    model_probability_by_name: dict[str, float]
    independent_edge_probability_by_name: dict[str, float]
    relative_to_independent_edges_by_name: dict[str, float]
    independent_dyad_probability_by_name: dict[str, float]
    relative_to_independent_dyads_by_name: dict[str, float]
    n_nodes: Optional[int]
    n_triplets: int
    total_possible_triplets: Optional[int]
    sampling: str
    include_empty: bool
    doublet_baseline: str


# Six-bit edge codes use the same bit order as triads._EDGE_POSITIONS.
_EDGE_MASKS = (
    (np.arange(64, dtype=np.uint8)[:, None] >> np.arange(6, dtype=np.uint8))
    & 1
).astype(bool)


def _triad_name_for_code(code: int) -> str:
    outgoing = np.zeros((3, 3), dtype=bool)
    for bit, (source, target) in enumerate(_EDGE_POSITIONS):
        outgoing[source, target] = bool((code >> bit) & 1)
    return _classify_outgoing_triad(outgoing)


_CODE_TRIAD_NAMES = tuple(_triad_name_for_code(code) for code in range(64))
_TRIAD_INDEX = {name: index for index, name in enumerate(TRIAD_NAMES)}
_CODE_TO_TRIAD_INDEX = np.asarray(
    [_TRIAD_INDEX[name] for name in _CODE_TRIAD_NAMES], dtype=np.int64
)
_CODE_TO_TRIAD_ONE_HOT = np.zeros((64, len(TRIAD_NAMES)), dtype=np.float64)
_CODE_TO_TRIAD_ONE_HOT[
    np.arange(64), _CODE_TO_TRIAD_INDEX
] = 1.0


def _validate_probability_matrix(P: Any) -> tuple[Any, int]:
    matrix, n_nodes = prepare_adjacency(P)
    values = matrix.data if is_sparse_matrix(matrix) else np.asarray(matrix)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("P must contain connection probabilities in [0, 1].")
    if n_nodes < 3:
        raise ValueError("P must describe at least three nodes.")
    return matrix, n_nodes


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _validate_triplets(triplets: Any, n_nodes: int) -> np.ndarray:
    raw = np.asarray(triplets)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("triplets must have shape (n_triplets, 3).")
    if raw.shape[0] == 0:
        raise ValueError("triplets must contain at least one triplet.")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("triplets must contain integer node indices.")
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("triplets must contain integer node indices.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("triplets contains NaN or infinite indices.")
    if not np.all(numeric == np.floor(numeric)):
        raise TypeError("triplets must contain integer node indices.")
    array = numeric.astype(np.int64)
    if np.any(array < 0) or np.any(array >= n_nodes):
        raise ValueError("triplets contains a node index outside P.")
    if np.any(
        (array[:, 0] == array[:, 1])
        | (array[:, 0] == array[:, 2])
        | (array[:, 1] == array[:, 2])
    ):
        raise ValueError("Each triplet must contain three distinct nodes.")
    return array


def _validate_doublet_baseline(value: str) -> DoubletBaseline:
    if value not in ("pooled", "position_specific"):
        raise ValueError(
            "doublet_baseline must be 'pooled' or 'position_specific'."
        )
    return value  # type: ignore[return-value]


def _unrank_unordered_triplets(
    n_nodes: int,
    ranks: np.ndarray,
) -> np.ndarray:
    """Map lexicographic combination ranks to unordered node triplets."""
    ranks = np.asarray(ranks, dtype=np.int64)
    total = comb(n_nodes, 3)
    first_nodes = np.arange(n_nodes - 2, dtype=np.int64)
    remaining = n_nodes - first_nodes
    first_starts = total - (remaining * (remaining - 1) * (remaining - 2) // 6)
    first = np.searchsorted(first_starts, ranks, side="right") - 1
    first = first.astype(np.int64, copy=False)
    residual = ranks - first_starts[first]

    # With the first node fixed, unrank a two-element combination from the
    # m = n_nodes - first - 1 remaining nodes.  The lexicographic start for
    # second-node offset j is j * (2*m - j - 1) / 2.
    m = n_nodes - first - 1
    coefficient = 2 * m - 1
    discriminant = np.maximum(
        coefficient.astype(np.float64) ** 2 - 8.0 * residual, 0.0
    )
    offset = np.floor(
        0.5 * (coefficient - np.sqrt(discriminant))
    ).astype(np.int64)
    offset = np.clip(offset, 0, m - 2)
    start = offset * (2 * m - offset - 1) // 2

    # Correct any boundary rounding from the floating-point square root.
    too_large = start > residual
    while np.any(too_large):
        offset[too_large] -= 1
        start = offset * (2 * m - offset - 1) // 2
        too_large = start > residual
    next_start = (offset + 1) * (2 * m - offset - 2) // 2
    too_small = (offset < m - 2) & (next_start <= residual)
    while np.any(too_small):
        offset[too_small] += 1
        start = offset * (2 * m - offset - 1) // 2
        next_start = (offset + 1) * (2 * m - offset - 2) // 2
        too_small = (offset < m - 2) & (next_start <= residual)

    second = first + 1 + offset
    third = second + 1 + (residual - start)
    return np.column_stack((first, second, third)).astype(np.int64, copy=False)


def _sample_unordered_triplets(
    n_nodes: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unordered three-node subsets uniformly, with replacement."""
    total = comb(n_nodes, 3)
    if total > np.iinfo(np.int64).max:
        raise ValueError("The number of possible triplets exceeds int64 sampling.")
    ranks = rng.integers(0, total, size=n_samples, dtype=np.int64)
    return _unrank_unordered_triplets(n_nodes, ranks)


def _triplet_batches(
    *,
    n_nodes: int,
    triplets: Optional[np.ndarray],
    sample_size: Optional[int],
    chunk_size: int,
    random_state: Any,
    max_exact_triplets: Optional[int],
) -> tuple[Iterator[np.ndarray], int, str]:
    total_possible = comb(n_nodes, 3)
    if triplets is not None:
        n_triplets = int(triplets.shape[0])

        def explicit_batches() -> Iterator[np.ndarray]:
            for start in range(0, n_triplets, chunk_size):
                yield triplets[start : start + chunk_size]

        return explicit_batches(), n_triplets, "explicit_ordered_triplets"

    if sample_size is not None:
        sample_size = _validate_positive_integer(
            sample_size, name="sample_size"
        )
        rng = np.random.default_rng(random_state)

        def sampled_batches() -> Iterator[np.ndarray]:
            remaining = sample_size
            while remaining > 0:
                current = min(chunk_size, remaining)
                yield _sample_unordered_triplets(n_nodes, current, rng)
                remaining -= current

        return (
            sampled_batches(),
            sample_size,
            "uniform_unordered_triplets_with_replacement",
        )

    if max_exact_triplets is not None:
        max_exact_triplets = _validate_positive_integer(
            max_exact_triplets, name="max_exact_triplets"
        )
        if total_possible > max_exact_triplets:
            raise ValueError(
                f"Exact enumeration requires {total_possible:,} triplets, "
                f"which exceeds max_exact_triplets={max_exact_triplets:,}. "
                "Set sample_size for Monte Carlo sampling, provide explicit "
                "triplets, or increase max_exact_triplets."
            )

    def exact_batches() -> Iterator[np.ndarray]:
        iterator = combinations(range(n_nodes), 3)
        while True:
            batch = list(islice(iterator, chunk_size))
            if not batch:
                break
            yield np.asarray(batch, dtype=np.int64)

    return exact_batches(), total_possible, "all_unordered_triplets"


def _matrix_entries(matrix: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    values = matrix[rows, cols]
    if is_sparse_matrix(matrix):
        return np.asarray(values).reshape(-1).astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _edge_probabilities_for_triplets(
    matrix: Any,
    triplets: np.ndarray,
) -> np.ndarray:
    node0 = triplets[:, 0]
    node1 = triplets[:, 1]
    node2 = triplets[:, 2]
    # P[target, source] for edge source -> target.
    return np.column_stack(
        (
            _matrix_entries(matrix, node1, node0),  # 0 -> 1
            _matrix_entries(matrix, node0, node1),  # 1 -> 0
            _matrix_entries(matrix, node2, node0),  # 0 -> 2
            _matrix_entries(matrix, node0, node2),  # 2 -> 0
            _matrix_entries(matrix, node2, node1),  # 1 -> 2
            _matrix_entries(matrix, node1, node2),  # 2 -> 1
        )
    )


def _pattern_probabilities(edge_probabilities: np.ndarray) -> np.ndarray:
    """Return probabilities of all 64 labeled six-edge patterns."""
    probabilities = np.ones((edge_probabilities.shape[0], 1), dtype=np.float64)
    for edge_index in range(6):
        edge = edge_probabilities[:, edge_index : edge_index + 1]
        probabilities = np.concatenate(
            (probabilities * (1.0 - edge), probabilities * edge), axis=1
        )
    return probabilities


def _class_probabilities_standard_order(
    edge_probabilities: np.ndarray,
) -> np.ndarray:
    return _pattern_probabilities(edge_probabilities) @ _CODE_TO_TRIAD_ONE_HOT


def _dyad_state_probabilities(edge_probabilities: np.ndarray) -> np.ndarray:
    """Return absent/forward/reverse/reciprocal probabilities for 3 dyads."""
    result = np.empty((edge_probabilities.shape[0], 3, 4), dtype=np.float64)
    for dyad_index, first_edge in enumerate((0, 2, 4)):
        forward = edge_probabilities[:, first_edge]
        reverse = edge_probabilities[:, first_edge + 1]
        result[:, dyad_index, 0] = (1.0 - forward) * (1.0 - reverse)
        result[:, dyad_index, 1] = forward * (1.0 - reverse)
        result[:, dyad_index, 2] = (1.0 - forward) * reverse
        result[:, dyad_index, 3] = forward * reverse
    return result


def _class_probabilities_from_dyad_states(
    dyad_state_probabilities: np.ndarray,
) -> np.ndarray:
    """Combine three independent labeled dyads into triad classes."""
    if dyad_state_probabilities.shape != (3, 4):
        raise ValueError("dyad_state_probabilities must have shape (3, 4).")
    pattern_probabilities = np.ones(64, dtype=np.float64)
    for dyad_index, first_edge in enumerate((0, 2, 4)):
        forward_bit = _EDGE_MASKS[:, first_edge].astype(np.int64)
        reverse_bit = _EDGE_MASKS[:, first_edge + 1].astype(np.int64)
        state = forward_bit + 2 * reverse_bit
        pattern_probabilities *= dyad_state_probabilities[dyad_index, state]
    return pattern_probabilities @ _CODE_TO_TRIAD_ONE_HOT


def _paper_order(include_empty: bool) -> tuple[str, ...]:
    if include_empty:
        return ("003",) + UDVARY_TRIPLET_MOTIF_NAMES
    return UDVARY_TRIPLET_MOTIF_NAMES


def _reorder_standard(
    values: np.ndarray,
    *,
    include_empty: bool,
) -> np.ndarray:
    order = _paper_order(include_empty)
    indices = [_TRIAD_INDEX[name] for name in order]
    return np.asarray(values)[..., indices]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    positive = denominator > 0.0
    np.divide(numerator, denominator, out=result, where=positive)
    result[(denominator == 0.0) & (numerator > 0.0)] = np.inf
    return result


def _log2_ratio(values: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(values)


def _prepare_edge_probability_rows(
    edge_probabilities: Any,
) -> tuple[np.ndarray, bool]:
    """Validate six-edge probability rows and report whether input was 1-D."""
    raw = np.asarray(edge_probabilities)
    was_one_dimensional = raw.ndim == 1
    if was_one_dimensional:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("edge_probabilities must have shape (6,) or (n, 6).")
    if raw.shape[0] == 0:
        raise ValueError("edge_probabilities must contain at least one triplet.")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("Complex-valued probabilities are not supported.")
    try:
        probabilities = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("edge_probabilities must contain numeric values.") from exc
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("edge_probabilities contains NaN or infinite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("edge_probabilities must lie in [0, 1].")
    return probabilities, was_one_dimensional


def triplet_motif_class_probabilities(
    edge_probabilities: Any,
    *,
    include_empty: bool = False,
) -> np.ndarray:
    """Calculate induced triad-class probabilities for six edge probabilities.

    Parameters
    ----------
    edge_probabilities:
        Array with shape ``(6,)`` or ``(n_triplets, 6)`` in the edge order
        ``0->1, 1->0, 0->2, 2->0, 1->2, 2->1``.
    include_empty:
        Include the empty ``003`` triad as motif ID 0.  By default, return the
        15 nonempty motifs in the visual order used by Udvary et al.

    Returns
    -------
    numpy.ndarray
        Shape ``(15,)``/``(16,)`` for one input row, or
        ``(n_triplets, 15)``/``(n_triplets, 16)`` for multiple rows.
    """
    if not isinstance(include_empty, (bool, np.bool_)):
        raise TypeError("include_empty must be a Boolean value.")
    probabilities, was_one_dimensional = _prepare_edge_probability_rows(
        edge_probabilities
    )
    result = _reorder_standard(
        _class_probabilities_standard_order(probabilities),
        include_empty=include_empty,
    )
    return result[0] if was_one_dimensional else result


def _summarize_edge_probability_batches(
    edge_batches: Iterator[np.ndarray],
    *,
    n_triplets: int,
    n_nodes: Optional[int],
    total_possible_triplets: Optional[int],
    sampling: str,
    include_empty: bool,
    doublet_baseline: DoubletBaseline,
) -> TripletMotifProbabilityResult:
    """Accumulate model and null probabilities from batches of six-edge rows."""
    class_sum = np.zeros(len(TRIAD_NAMES), dtype=np.float64)
    class_square_sum = np.zeros(len(TRIAD_NAMES), dtype=np.float64)
    edge_sum = np.zeros(6, dtype=np.float64)
    dyad_sum = np.zeros((3, 4), dtype=np.float64)
    processed = 0

    for edge_probabilities in edge_batches:
        if edge_probabilities.ndim != 2 or edge_probabilities.shape[1] != 6:
            raise RuntimeError("Internal edge-probability batch shape mismatch.")
        class_probabilities = _class_probabilities_standard_order(
            edge_probabilities
        )
        dyad_probabilities = _dyad_state_probabilities(edge_probabilities)
        class_sum += class_probabilities.sum(axis=0)
        class_square_sum += np.square(class_probabilities).sum(axis=0)
        edge_sum += edge_probabilities.sum(axis=0)
        dyad_sum += dyad_probabilities.sum(axis=0)
        processed += edge_probabilities.shape[0]

    if processed != n_triplets:  # pragma: no cover - internal invariant.
        raise RuntimeError("Internal triplet batching count mismatch.")

    model_probability_standard = class_sum / n_triplets
    if n_triplets > 1:
        variance = (
            class_square_sum
            - n_triplets * np.square(model_probability_standard)
        ) / (n_triplets - 1)
        variance = np.maximum(variance, 0.0)
        model_sem_standard = np.sqrt(variance / n_triplets)
    else:
        model_sem_standard = np.zeros(len(TRIAD_NAMES), dtype=np.float64)

    mean_edge_probability = edge_sum / n_triplets
    mean_dyad_state_probability = dyad_sum / n_triplets

    # Udvary et al. direct random-network baseline: replace each of the six
    # edge positions by its across-triplet mean, then recompute motif
    # probabilities analytically.
    edge_random_standard = _class_probabilities_standard_order(
        mean_edge_probability[None, :]
    )[0]
    edge_random_dyad = _dyad_state_probabilities(
        mean_edge_probability[None, :]
    )[0]

    # Song et al. / Udvary Fig. 6E baseline: preserve measured/predicted
    # absent, unidirectional, and reciprocal dyad frequencies, but combine
    # the three dyads independently.  The pooled unidirectional probability
    # is split equally between the two possible directions.
    pooled_dyad = np.asarray(
        [
            mean_dyad_state_probability[:, 0].mean(),
            0.5
            * (
                mean_dyad_state_probability[:, 1]
                + mean_dyad_state_probability[:, 2]
            ).mean(),
            0.5
            * (
                mean_dyad_state_probability[:, 1]
                + mean_dyad_state_probability[:, 2]
            ).mean(),
            mean_dyad_state_probability[:, 3].mean(),
        ],
        dtype=np.float64,
    )
    pooled_edge_random_dyad = np.asarray(
        [
            edge_random_dyad[:, 0].mean(),
            0.5 * (edge_random_dyad[:, 1] + edge_random_dyad[:, 2]).mean(),
            0.5 * (edge_random_dyad[:, 1] + edge_random_dyad[:, 2]).mean(),
            edge_random_dyad[:, 3].mean(),
        ],
        dtype=np.float64,
    )

    if doublet_baseline == "pooled":
        baseline_dyads = np.repeat(pooled_dyad[None, :], 3, axis=0)
    else:
        baseline_dyads = mean_dyad_state_probability
    dyad_random_standard = _class_probabilities_from_dyad_states(
        baseline_dyads
    )

    order = _paper_order(include_empty)
    motif_ids = (
        np.arange(0, 16, dtype=np.int64)
        if include_empty
        else np.arange(1, 16, dtype=np.int64)
    )
    model_probability = _reorder_standard(
        model_probability_standard, include_empty=include_empty
    )
    model_sem = _reorder_standard(
        model_sem_standard, include_empty=include_empty
    )
    edge_random_probability = _reorder_standard(
        edge_random_standard, include_empty=include_empty
    )
    dyad_random_probability = _reorder_standard(
        dyad_random_standard, include_empty=include_empty
    )
    edge_ratio = _safe_ratio(model_probability, edge_random_probability)
    dyad_ratio = _safe_ratio(model_probability, dyad_random_probability)
    doublet_relative = _safe_ratio(
        mean_dyad_state_probability, edge_random_dyad
    )
    pooled_doublet_relative = _safe_ratio(
        pooled_dyad, pooled_edge_random_dyad
    )

    model_by_name = dict(zip(order, map(float, model_probability)))
    edge_probability_by_name = dict(
        zip(order, map(float, edge_random_probability))
    )
    edge_ratio_by_name = dict(zip(order, map(float, edge_ratio)))
    dyad_probability_by_name = dict(
        zip(order, map(float, dyad_random_probability))
    )
    dyad_ratio_by_name = dict(zip(order, map(float, dyad_ratio)))

    return {
        "motif_id": motif_ids,
        "triad": np.asarray(order),
        "model_probability": model_probability,
        "model_probability_standard_error": model_sem,
        "independent_edge_probability": edge_random_probability,
        "relative_to_independent_edges": edge_ratio,
        "log2_relative_to_independent_edges": _log2_ratio(edge_ratio),
        "independent_dyad_probability": dyad_random_probability,
        "relative_to_independent_dyads": dyad_ratio,
        "doublet_normalized_ratio": dyad_ratio.copy(),
        "log2_relative_to_independent_dyads": _log2_ratio(dyad_ratio),
        "mean_edge_probability": mean_edge_probability,
        "edge_position": np.asarray(TRIPLET_EDGE_NAMES),
        "mean_dyad_state_probability": mean_dyad_state_probability,
        "independent_edge_dyad_state_probability": edge_random_dyad,
        "doublet_relative_occurrence": doublet_relative,
        "pooled_dyad_state_probability": pooled_dyad,
        "pooled_independent_edge_dyad_state_probability": pooled_edge_random_dyad,
        "pooled_doublet_relative_occurrence": pooled_doublet_relative,
        "dyad_position": np.asarray(TRIPLET_DYAD_NAMES),
        "dyad_state": np.asarray(TRIPLET_DYAD_STATE_NAMES),
        "model_probability_by_name": model_by_name,
        "independent_edge_probability_by_name": edge_probability_by_name,
        "relative_to_independent_edges_by_name": edge_ratio_by_name,
        "independent_dyad_probability_by_name": dyad_probability_by_name,
        "relative_to_independent_dyads_by_name": dyad_ratio_by_name,
        "n_nodes": n_nodes,
        "n_triplets": n_triplets,
        "total_possible_triplets": total_possible_triplets,
        "sampling": sampling,
        "include_empty": bool(include_empty),
        "doublet_baseline": doublet_baseline,
    }


def triplet_motif_probability_ratios_from_edge_probabilities(
    edge_probabilities: Any,
    *,
    chunk_size: int = 50_000,
    include_empty: bool = False,
    doublet_baseline: DoubletBaseline = "pooled",
) -> TripletMotifProbabilityResult:
    """Calculate Udvary/Song motif ratios from pre-extracted six-edge rows.

    This is the matrix-shape-independent entry point.  It is useful when the
    three node roles come from different populations represented by separate
    rectangular connectivity blocks.  Assemble one row per ordered triplet in
    the edge order

    ``0->1, 1->0, 0->2, 2->0, 1->2, 2->1``.

    The returned probabilities and random-network ratios are identical to
    :func:`triplet_motif_probability_ratios` when the rows were extracted from
    the same square probability matrix and ordered triplets.
    """
    if not isinstance(include_empty, (bool, np.bool_)):
        raise TypeError("include_empty must be a Boolean value.")
    chunk_size = _validate_positive_integer(chunk_size, name="chunk_size")
    doublet_baseline = _validate_doublet_baseline(doublet_baseline)
    probabilities, _ = _prepare_edge_probability_rows(edge_probabilities)
    n_triplets = int(probabilities.shape[0])

    def batches() -> Iterator[np.ndarray]:
        for start in range(0, n_triplets, chunk_size):
            yield probabilities[start : start + chunk_size]

    return _summarize_edge_probability_batches(
        batches(),
        n_triplets=n_triplets,
        n_nodes=None,
        total_possible_triplets=None,
        sampling="provided_edge_probability_rows",
        include_empty=bool(include_empty),
        doublet_baseline=doublet_baseline,
    )


def triplet_motif_probability_ratios(
    P: Any,
    *,
    triplets: Any = None,
    sample_size: Optional[int] = None,
    random_state: Any = None,
    chunk_size: int = 50_000,
    max_exact_triplets: Optional[int] = 2_000_000,
    include_empty: bool = False,
    doublet_baseline: DoubletBaseline = "pooled",
) -> TripletMotifProbabilityResult:
    """Estimate triplet-motif probabilities and random-network ratios.

    ``P`` is a matrix of *connection probabilities*, not a realized binary
    adjacency matrix.  For every selected triplet, this function computes the
    expected probability of each induced directed-triad class from its six
    directed edge probabilities and averages across triplets.

    Parameters
    ----------
    P:
        Square connection-probability matrix with entries in ``[0, 1]`` and
        convention ``P[target, source]``.
    triplets:
        Optional integer array of shape ``(n_triplets, 3)``.  Row order is
        preserved and defines positions 0, 1, and 2.  This is important for
        cell-type-specific triplets, where the six edge-position means can be
        different.  Duplicate triplet rows are allowed.
    sample_size:
        If ``triplets`` is omitted, sample this many unordered three-node sets
        uniformly with replacement.  This is a scalable analogue of the
        eight-million-triplet sampling used by Udvary et al.  If both
        ``triplets`` and ``sample_size`` are omitted, enumerate all unordered
        triples exactly.
    random_state:
        Seed or ``numpy.random.Generator`` used only when ``sample_size`` is
        provided.
    chunk_size:
        Number of triplets processed at once.  Results are accumulated without
        retaining all per-triplet motif probabilities.
    max_exact_triplets:
        Safety limit for exact enumeration.  Set to ``None`` to disable it.
    include_empty:
        Include the empty triad ``003``.  The paper plots only 15 nonempty
        motifs, so the default is ``False``.
    doublet_baseline:
        ``"pooled"`` reproduces the Song-style homogeneous-population null:
        absent and reciprocal doublet frequencies are pooled across the three
        node pairs, and the two one-way orientations share the pooled
        unidirectional probability equally. ``"position_specific"`` preserves
        all four labeled dyad-state means separately for each node-pair
        position; this is useful for ordered heterogeneous cell types.

    Returns
    -------
    TripletMotifProbabilityResult
        ``relative_to_independent_edges`` is the Fig. 3E-style ratio.
        ``relative_to_independent_dyads`` (also returned as
        ``doublet_normalized_ratio``) is the Song/Fig. 6E-style ratio after
        accounting for doublet statistics.

    Notes
    -----
    For a realized binary network, use :func:`directed_triad_census` or
    :func:`triad_enrichment` instead.  Those functions count observed
    subgraphs; this function averages probabilities across an ensemble.

    For separate rectangular population-to-population matrices, first extract
    the six directed probabilities for every ordered triplet and call
    :func:`triplet_motif_probability_ratios_from_edge_probabilities`.
    """
    if triplets is not None and sample_size is not None:
        raise ValueError("Specify at most one of triplets and sample_size.")
    if not isinstance(include_empty, (bool, np.bool_)):
        raise TypeError("include_empty must be a Boolean value.")
    chunk_size = _validate_positive_integer(chunk_size, name="chunk_size")
    doublet_baseline = _validate_doublet_baseline(doublet_baseline)
    matrix, n_nodes = _validate_probability_matrix(P)
    validated_triplets = (
        None if triplets is None else _validate_triplets(triplets, n_nodes)
    )
    triplet_batches, n_triplets, sampling = _triplet_batches(
        n_nodes=n_nodes,
        triplets=validated_triplets,
        sample_size=sample_size,
        chunk_size=chunk_size,
        random_state=random_state,
        max_exact_triplets=max_exact_triplets,
    )

    def edge_batches() -> Iterator[np.ndarray]:
        for batch in triplet_batches:
            yield _edge_probabilities_for_triplets(matrix, batch)

    return _summarize_edge_probability_batches(
        edge_batches(),
        n_triplets=n_triplets,
        n_nodes=n_nodes,
        total_possible_triplets=comb(n_nodes, 3),
        sampling=sampling,
        include_empty=bool(include_empty),
        doublet_baseline=doublet_baseline,
    )

def udvary_triplet_motif_probability_ratios(
    P: Any,
    **kwargs: Any,
) -> TripletMotifProbabilityResult:
    """Alias emphasizing the Udvary et al. triplet-motif analysis."""
    return triplet_motif_probability_ratios(P, **kwargs)
