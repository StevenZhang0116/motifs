"""Udvary/Song-style expected triplet-motif probability ratios."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow this example to run directly from an unpacked source directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motif_cumulants import triplet_motif_probability_ratios


EDGE_POSITIONS = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
)


def assign_triplet(
    P: np.ndarray,
    nodes: tuple[int, int, int],
    edge_probabilities: np.ndarray,
) -> None:
    """Assign six directed probabilities using P[target, source]."""
    for probability, (source_position, target_position) in zip(
        edge_probabilities, EDGE_POSITIONS
    ):
        source = nodes[source_position]
        target = nodes[target_position]
        P[target, source] = probability


# Two triplets have the same within-triplet edge profile but very different
# overall density. Averaging motif probabilities before averaging edge
# probabilities makes dense and empty structures more likely than in an
# independent-edge random network.
P = np.zeros((6, 6), dtype=float)
assign_triplet(P, (0, 1, 2), np.full(6, 0.1))
assign_triplet(P, (3, 4, 5), np.full(6, 0.9))
ordered_triplets = np.array([[0, 1, 2], [3, 4, 5]])

result = triplet_motif_probability_ratios(
    P,
    triplets=ordered_triplets,
    doublet_baseline="pooled",
)

print(
    "motif  model probability  edge-random ratio  "
    "doublet-normalized ratio"
)
for name, probability, edge_ratio, doublet_ratio in zip(
    result["triad"],
    result["model_probability"],
    result["relative_to_independent_edges"],
    result["doublet_normalized_ratio"],
):
    print(
        f"{name:>5s}  {probability:17.8g}  "
        f"{edge_ratio:17.8g}  {doublet_ratio:24.8g}"
    )

print("\nMean six-edge probabilities:")
for edge, probability in zip(
    result["edge_position"], result["mean_edge_probability"]
):
    print(f"  {edge}: {probability:.3f}")
