"""Classical induced directed-triad census.

Motif cumulants count weighted walks and may reuse nodes. A classical triad
census answers a different question: each unordered set of three *distinct*
nodes is assigned to exactly one of the 16 directed triad isomorphism classes.
This module keeps that subgraph analysis separate from the analytic cumulant
formulas.

The class names follow the standard MAN notation:

``003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201,
120D, 120U, 120C, 210, 300``.

Two census modes are available via the ``sample_size`` parameter of
:func:`directed_triad_census`:

* **Exact** (``sample_size=None``, default): enumerate all C(N, 3) unordered
  triples in O(N³) time.  Counts are exact integers.
* **Monte Carlo** (``sample_size=S``): draw S random triples uniformly, then
  classify each with a fully vectorised NumPy operation in O(S) time.  Counts
  are estimates with Monte Carlo noise of order 1 / √S.  A value around
  ``S = 200 000`` is sufficient for stable z-scores in typical networks
  (N ~ 100–300).

References
----------
Milo et al. (2002), *Network motifs: Simple building blocks of complex
networks*, Science 298, 824-827.
https://doi.org/10.1126/science.298.5594.824

Adjacency convention
--------------------
``W[i, j]`` represents ``j -> i``. Edge weights are thresholded to obtain a
binary topology; diagonal entries are ignored.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Literal, Optional, TypedDict

import numpy as np

from ._validation import is_sparse_matrix, prepare_adjacency


EdgeRule = Literal["absolute", "positive"]
TRIAD_NAMES = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
)

# Edge codes use the conventional outgoing orientation source -> target.
_EDGE_POSITIONS = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
)
_PERMUTATIONS = tuple(permutations(range(3)))
_TRIAD_INDEX_BY_NAME: dict = {name: idx for idx, name in enumerate(TRIAD_NAMES)}

# One representative outgoing-edge set for every triad class.
_REPRESENTATIVE_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "003": (),
    "012": ((0, 1),),
    "102": ((0, 1), (1, 0)),
    "021D": ((0, 1), (0, 2)),
    "021U": ((1, 0), (2, 0)),
    "021C": ((0, 1), (1, 2)),
    "111D": ((0, 1), (1, 0), (2, 0)),
    "111U": ((0, 1), (1, 0), (0, 2)),
    "030T": ((0, 1), (1, 2), (0, 2)),
    "030C": ((0, 1), (1, 2), (2, 0)),
    "201": ((0, 1), (1, 0), (0, 2), (2, 0)),
    "120D": ((0, 1), (1, 0), (2, 0), (2, 1)),
    "120U": ((0, 1), (1, 0), (0, 2), (1, 2)),
    "120C": ((0, 1), (1, 0), (1, 2), (2, 0)),
    "210": ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2)),
    "300": (
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ),
}


class DirectedTriadCensusResult(TypedDict):
    """Result returned by :func:`directed_triad_census`.

    When the census was run in Monte Carlo mode (``sample_size`` is not
    ``None``), ``counts`` contains estimated integers rounded from
    ``proportions × C(N, 3)`` and ``proportions`` are estimated class
    frequencies from the random sample.  In exact mode all values are exact.
    """

    triad: np.ndarray
    counts: np.ndarray
    fractions: np.ndarray
    proportions: np.ndarray
    count_by_name: dict
    proportion_by_name: dict
    total_triples: int
    n_nodes: int
    n_edges: int
    threshold: float
    edge_presence: str
    edge_threshold: float
    edge_rule: str
    sample_size: Optional[int]


def _validate_threshold(edge_threshold: float) -> float:
    if isinstance(edge_threshold, (bool, np.bool_)):
        raise TypeError("edge_threshold must be a nonnegative real number.")
    try:
        threshold = float(edge_threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "edge_threshold must be a nonnegative real number."
        ) from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("edge_threshold must be finite and nonnegative.")
    return threshold


def _validate_edge_rule(edge_rule: str) -> EdgeRule:
    if edge_rule not in ("absolute", "positive"):
        raise ValueError("edge_rule must be 'absolute' or 'positive'.")
    return edge_rule  # type: ignore[return-value]


def _outgoing_topology(
    W: Any,
    *,
    edge_threshold: float,
    edge_rule: EdgeRule,
) -> np.ndarray:
    matrix, _ = prepare_adjacency(W)
    if is_sparse_matrix(matrix):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)

    if edge_rule == "absolute":
        present = np.abs(dense) > edge_threshold
    else:
        present = dense > edge_threshold
    present = np.asarray(present, dtype=bool)
    np.fill_diagonal(present, False)

    # Convert package orientation W[target, source] to outgoing[source, target].
    return present.T


def _edge_code(outgoing: np.ndarray) -> int:
    code = 0
    for bit, (source, target) in enumerate(_EDGE_POSITIONS):
        if outgoing[source, target]:
            code |= 1 << bit
    return code


def _canonical_code(outgoing: np.ndarray) -> int:
    """Return the minimum edge code across all node relabelings."""
    return min(
        _edge_code(outgoing[np.ix_(permutation, permutation)])
        for permutation in _PERMUTATIONS
    )


def _representative_matrix(
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    outgoing = np.zeros((3, 3), dtype=bool)
    for source, target in edges:
        outgoing[source, target] = True
    return outgoing


_TRIAD_BY_CANONICAL_CODE = {
    _canonical_code(_representative_matrix(edges)): name
    for name, edges in _REPRESENTATIVE_EDGES.items()
}
if len(_TRIAD_BY_CANONICAL_CODE) != len(TRIAD_NAMES):  # pragma: no cover
    raise RuntimeError("Internal triad representatives are not unique.")


# ---------------------------------------------------------------------------
# Precomputed lookup table for vectorised Monte Carlo triple classification
# ---------------------------------------------------------------------------

def _build_raw_code_to_triad_idx() -> np.ndarray:
    """Return a 64-entry lookup table mapping any 6-bit edge code to a triad index.

    The 6 bits correspond to the 6 possible directed edges between 3 labelled
    nodes, in the order defined by ``_EDGE_POSITIONS``::

        bit 0: node-0 → node-1   bit 1: node-1 → node-0
        bit 2: node-0 → node-2   bit 3: node-2 → node-0
        bit 4: node-1 → node-2   bit 5: node-2 → node-1

    Because the canonical code is the minimum over all node relabelings, the
    same triad class is returned regardless of which of the 6 orderings of the
    three nodes was used to assign bits 0–5.
    """
    table = np.empty(64, dtype=np.uint8)
    for raw in range(64):
        outgoing = np.zeros((3, 3), dtype=bool)
        for bit, (src, tgt) in enumerate(_EDGE_POSITIONS):
            if raw & (1 << bit):
                outgoing[src, tgt] = True
        canon = _canonical_code(outgoing)
        name = _TRIAD_BY_CANONICAL_CODE[canon]
        table[raw] = _TRIAD_INDEX_BY_NAME[name]
    return table


_RAW_CODE_TO_TRIAD_IDX: np.ndarray = _build_raw_code_to_triad_idx()

# Powers-of-two used to convert a (S, 6) bool/uint8 matrix to S integer codes.
_CLASSIFY_POWERS: np.ndarray = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)


def _rng_from_random_state(random_state: object) -> np.random.Generator:
    """Resolve an integer seed, a Generator, or None to a numpy Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _sample_random_triples(
    n_nodes: int,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a ``(sample_size, 3)`` array of random unordered node triples.

    Each row contains three distinct indices in sorted order (i < j < k),
    drawn uniformly with replacement from the C(``n_nodes``, 3) possible
    unordered triples.

    Parameters
    ----------
    n_nodes:
        Number of nodes; must be ≥ 3.
    sample_size:
        Number of triples to return.
    rng:
        NumPy random generator.

    Returns
    -------
    ndarray of shape (sample_size, 3), dtype int64
    """
    if n_nodes < 3:
        raise ValueError(
            f"_sample_random_triples requires n_nodes >= 3; got {n_nodes}."
        )
    parts = []
    remaining = sample_size
    while remaining > 0:
        # Oversample to compensate for the ~3/n collision rate after dedup.
        n_draw = remaining + max(16, remaining // 8)
        raw = rng.integers(0, n_nodes, size=(n_draw, 3), dtype=np.int64)
        raw.sort(axis=1)
        valid = (raw[:, 0] != raw[:, 1]) & (raw[:, 1] != raw[:, 2])
        good = raw[valid]
        if len(good) > remaining:
            good = good[:remaining]
        parts.append(good)
        remaining -= len(good)
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _classify_triples_vectorized(
    outgoing: np.ndarray,
    triples: np.ndarray,
) -> np.ndarray:
    """Classify an array of node triples into triad class indices.

    The classification is fully vectorised: no Python loop over triples.

    Parameters
    ----------
    outgoing:
        ``(n, n)`` bool array; ``outgoing[i, j]`` is True when there is a
        directed edge i → j.
    triples:
        ``(S, 3)`` int array.  Each row ``[a, b, c]`` assigns node ``a`` to
        position 0, ``b`` to position 1, ``c`` to position 2 in the
        ``_EDGE_POSITIONS`` bit layout.  Any consistent ordering works because
        ``_RAW_CODE_TO_TRIAD_IDX`` already accounts for all node relabelings.

    Returns
    -------
    (S,) uint8 ndarray of indices into :data:`TRIAD_NAMES`.
    """
    a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    # Extract the 6 directed edge indicators in _EDGE_POSITIONS order.
    bits = np.stack([
        outgoing[a, b],   # bit 0: position-0 → position-1
        outgoing[b, a],   # bit 1: position-1 → position-0
        outgoing[a, c],   # bit 2: position-0 → position-2
        outgoing[c, a],   # bit 3: position-2 → position-0
        outgoing[b, c],   # bit 4: position-1 → position-2
        outgoing[c, b],   # bit 5: position-2 → position-1
    ], axis=1).astype(np.int32)                    # (S, 6), values 0 or 1
    codes = bits @ _CLASSIFY_POWERS               # (S,) int32, range 0–63
    return _RAW_CODE_TO_TRIAD_IDX[codes]


def _classify_outgoing_triad(outgoing: np.ndarray) -> str:
    try:
        return _TRIAD_BY_CANONICAL_CODE[_canonical_code(outgoing)]
    except KeyError as exc:  # pragma: no cover; all 2^6 patterns are covered.
        raise RuntimeError("Unrecognized directed triad topology.") from exc


def _resolve_edge_options(
    *,
    threshold: float,
    edge_presence: str,
    edge_threshold: Optional[float],
    edge_rule: Optional[EdgeRule],
) -> tuple[float, EdgeRule, str]:
    """Resolve old and new threshold option names without ambiguity."""
    if edge_threshold is not None:
        if threshold != 0.0:
            raise ValueError(
                "Specify only one of threshold and edge_threshold."
            )
        threshold = edge_threshold
    threshold = _validate_threshold(threshold)

    if edge_presence not in ("nonzero", "positive"):
        raise ValueError("edge_presence must be 'nonzero' or 'positive'.")
    mapped_rule: EdgeRule = (
        "absolute" if edge_presence == "nonzero" else "positive"
    )
    if edge_rule is not None:
        edge_rule = _validate_edge_rule(edge_rule)
        if edge_presence != "nonzero" and edge_rule != mapped_rule:
            raise ValueError(
                "edge_presence and edge_rule specify conflicting rules."
            )
        mapped_rule = edge_rule
    canonical_presence = (
        "nonzero" if mapped_rule == "absolute" else "positive"
    )
    return threshold, mapped_rule, canonical_presence


def classify_directed_triad(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
    edge_threshold: Optional[float] = None,
    edge_rule: Optional[EdgeRule] = None,
) -> str:
    """Classify one three-node adjacency matrix into a directed triad class.

    ``threshold``/``edge_presence`` are the primary public names.
    ``edge_threshold``/``edge_rule`` are accepted as descriptive aliases.
    Self-loops are ignored. ``edge_presence='nonzero'`` uses absolute weight;
    ``'positive'`` ignores negative connections.
    """
    threshold, resolved_rule, _ = _resolve_edge_options(
        threshold=threshold,
        edge_presence=edge_presence,
        edge_threshold=edge_threshold,
        edge_rule=edge_rule,
    )
    outgoing = _outgoing_topology(
        W,
        edge_threshold=threshold,
        edge_rule=resolved_rule,
    )
    if outgoing.shape != (3, 3):
        raise ValueError("classify_directed_triad requires a 3 x 3 matrix.")
    return _classify_outgoing_triad(outgoing)


def directed_triad_census(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
    edge_threshold: Optional[float] = None,
    edge_rule: Optional[EdgeRule] = None,
    sample_size: Optional[int] = None,
    random_state: Any = None,
) -> DirectedTriadCensusResult:
    """Count all 16 induced directed triad classes.

    Every unordered three-node subset contributes exactly once.

    Parameters
    ----------
    W:
        Square adjacency matrix.  ``W[i, j]`` = weight of edge j → i.
        Dense NumPy arrays and SciPy sparse matrices are both accepted.
    threshold:
        Edges with ``|weight| ≤ threshold`` (or ``weight ≤ threshold`` for
        ``edge_presence='positive'``) are treated as absent.  Default 0.
    edge_presence:
        ``'nonzero'`` tests ``|weight| > threshold``; ``'positive'`` tests
        ``weight > threshold``.  Default ``'nonzero'``.
    edge_threshold:
        Descriptive alias for ``threshold``.  Specify only one.
    edge_rule:
        Descriptive alias for ``edge_presence``.  Specify only one.
    sample_size:
        When ``None`` (default), all C(N, 3) triples are enumerated exactly in
        O(N³) time and ``counts`` are exact integers.  When a positive integer
        is given, that many random triples are drawn uniformly and classified in
        O(sample_size) time using a fully vectorised NumPy code path; ``counts``
        are then estimated integers (``round(proportions × C(N, 3))``) with
        Monte Carlo noise of order 1 / √sample_size.  A value around 200 000 is
        sufficient for stable z-scores in networks of N ~ 100–300.
    random_state:
        Integer seed, ``numpy.random.Generator``, or ``None``.  Used only when
        ``sample_size`` is given.

    Returns
    -------
    DirectedTriadCensusResult
        ``counts.sum()`` equals C(N, 3) for the exact path and approximately
        C(N, 3) for the Monte Carlo path.  ``sample_size`` records which mode
        was used (``None`` = exact).
    """
    threshold, resolved_rule, canonical_presence = _resolve_edge_options(
        threshold=threshold,
        edge_presence=edge_presence,
        edge_threshold=edge_threshold,
        edge_rule=edge_rule,
    )
    outgoing = _outgoing_topology(
        W,
        edge_threshold=threshold,
        edge_rule=resolved_rule,
    )
    n_nodes = outgoing.shape[0]
    n_edges = int(outgoing.sum())
    total = int(n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6)

    if sample_size is None:
        # --- Exact O(N³) path -------------------------------------------
        counts = np.zeros(len(TRIAD_NAMES), dtype=np.int64)
        for nodes in combinations(range(n_nodes), 3):
            subgraph = outgoing[np.ix_(nodes, nodes)]
            counts[_TRIAD_INDEX_BY_NAME[_classify_outgoing_triad(subgraph)]] += 1
        proportions = (
            counts.astype(np.float64) / total
            if total > 0
            else np.full(len(TRIAD_NAMES), np.nan, dtype=np.float64)
        )
    else:
        # --- Monte Carlo O(sample_size) path ----------------------------
        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("sample_size must be a positive integer.")
        if n_nodes < 3:
            # No valid triples exist; return zero estimates.
            counts = np.zeros(len(TRIAD_NAMES), dtype=np.int64)
            proportions = np.full(len(TRIAD_NAMES), np.nan, dtype=np.float64)
        else:
            rng = _rng_from_random_state(random_state)
            triples = _sample_random_triples(n_nodes, sample_size, rng)
            triad_indices = _classify_triples_vectorized(outgoing, triples)
            sample_counts = np.bincount(triad_indices, minlength=len(TRIAD_NAMES))
            proportions = sample_counts.astype(np.float64) / sample_size
            # Scale proportions to C(N,3) so counts are comparable to the
            # exact path in magnitude (important for z-score computation in
            # triad_enrichment, where observed and null are on the same scale).
            counts = np.round(proportions * total).astype(np.int64)

    count_by_name = {name: int(counts[idx]) for idx, name in enumerate(TRIAD_NAMES)}
    proportion_by_name = {
        name: float(proportions[idx]) for idx, name in enumerate(TRIAD_NAMES)
    }
    return {
        "triad": np.asarray(TRIAD_NAMES),
        "counts": counts,
        "fractions": proportions.copy(),
        "proportions": proportions,
        "count_by_name": count_by_name,
        "proportion_by_name": proportion_by_name,
        "total_triples": total,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "threshold": threshold,
        "edge_presence": canonical_presence,
        "edge_threshold": threshold,
        "edge_rule": resolved_rule,
        "sample_size": sample_size,
    }


def triad_census(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
    edge_threshold: Optional[float] = None,
    edge_rule: Optional[EdgeRule] = None,
    sample_size: Optional[int] = None,
    random_state: Any = None,
) -> DirectedTriadCensusResult:
    """Alias for :func:`directed_triad_census`."""
    return directed_triad_census(
        W,
        threshold=threshold,
        edge_presence=edge_presence,
        edge_threshold=edge_threshold,
        edge_rule=edge_rule,
        sample_size=sample_size,
        random_state=random_state,
    )
