"""Classical induced directed-triad census.

Motif cumulants count weighted walks and may reuse nodes. A classical triad
census answers a different question: each unordered set of three *distinct*
nodes is assigned to exactly one of the 16 directed triad isomorphism classes.
This module keeps that subgraph analysis separate from the analytic cumulant
formulas.

The class names follow the standard MAN notation:

``003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201,
120D, 120U, 120C, 210, 300``.

Reference
---------
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
    """Result returned by :func:`directed_triad_census`."""

    triad: np.ndarray
    counts: np.ndarray
    fractions: np.ndarray
    proportions: np.ndarray
    count_by_name: dict[str, int]
    proportion_by_name: dict[str, float]
    total_triples: int
    n_nodes: int
    n_edges: int
    threshold: float
    edge_presence: str
    edge_threshold: float
    edge_rule: str


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
) -> DirectedTriadCensusResult:
    """Count all 16 induced directed triad classes.

    Every unordered three-node subset contributes exactly once. Consequently,
    ``counts.sum() == comb(N, 3)``. The implementation is ``O(N**3)`` and is
    intended for networks where an exact triad census is computationally
    reasonable.
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
    counts = np.zeros(len(TRIAD_NAMES), dtype=np.int64)
    index_by_name = {name: index for index, name in enumerate(TRIAD_NAMES)}

    for nodes in combinations(range(n_nodes), 3):
        subgraph = outgoing[np.ix_(nodes, nodes)]
        counts[index_by_name[_classify_outgoing_triad(subgraph)]] += 1

    total = int(n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6)
    proportions = np.full(len(TRIAD_NAMES), np.nan, dtype=np.float64)
    if total > 0:
        proportions = counts.astype(np.float64) / total
    count_by_name = {
        name: int(counts[index]) for index, name in enumerate(TRIAD_NAMES)
    }
    proportion_by_name = {
        name: float(proportions[index])
        for index, name in enumerate(TRIAD_NAMES)
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
    }


def triad_census(
    W: Any,
    *,
    threshold: float = 0.0,
    edge_presence: str = "nonzero",
    edge_threshold: Optional[float] = None,
    edge_rule: Optional[EdgeRule] = None,
) -> DirectedTriadCensusResult:
    """Alias for :func:`directed_triad_census`."""
    return directed_triad_census(
        W,
        threshold=threshold,
        edge_presence=edge_presence,
        edge_threshold=edge_threshold,
        edge_rule=edge_rule,
    )
