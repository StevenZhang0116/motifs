"""Validation tests for exact finite-size SONET motif statistics."""

from __future__ import annotations

from itertools import combinations, permutations
import unittest

import numpy as np

from motif_cumulants import sonet_motif_statistics


def package_adjacency(
    edges: list[tuple[int, int]],
    n_nodes: int,
) -> np.ndarray:
    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for source, target in edges:
        W[target, source] = 1.0
    return W


def brute_force_counts(W: np.ndarray) -> np.ndarray:
    """Return chain/divergent/convergent/reciprocal distinct-node counts."""
    outgoing = W.T.astype(bool)
    n_nodes = W.shape[0]

    reciprocal = sum(
        int(outgoing[i, j] and outgoing[j, i])
        for i, j in combinations(range(n_nodes), 2)
    )
    divergent = 0
    convergent = 0
    chain = 0

    for center in range(n_nodes):
        other = [node for node in range(n_nodes) if node != center]
        divergent += sum(
            int(outgoing[center, first] and outgoing[center, second])
            for first, second in combinations(other, 2)
        )
        convergent += sum(
            int(outgoing[first, center] and outgoing[second, center])
            for first, second in combinations(other, 2)
        )

    for source, middle, target in permutations(range(n_nodes), 3):
        chain += int(
            outgoing[source, middle] and outgoing[middle, target]
        )

    return np.array(
        [chain, divergent, convergent, reciprocal], dtype=np.int64
    )


class SONETMotifTests(unittest.TestCase):
    def test_counts_match_brute_force_distinct_node_enumeration(self) -> None:
        W = package_adjacency(
            [
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 2),
            ],
            n_nodes=4,
        )
        result = sonet_motif_statistics(W)
        np.testing.assert_array_equal(result["counts"], brute_force_counts(W))
        np.testing.assert_array_equal(result["possible"], [24, 12, 12, 6])
        np.testing.assert_allclose(
            result["frequencies"],
            result["counts"] / result["possible"],
        )

    def test_complete_directed_graph_has_unit_frequencies(self) -> None:
        W = np.ones((5, 5), dtype=float) - np.eye(5)
        result = sonet_motif_statistics(W)
        self.assertAlmostEqual(result["connection_probability"], 1.0)
        np.testing.assert_allclose(result["frequencies"], 1.0)
        np.testing.assert_allclose(result["independent_edge_excess"], 0.0)
        np.testing.assert_allclose(result["alpha"], 0.0)

    def test_alpha_is_relative_excess_over_p_squared(self) -> None:
        W = package_adjacency(
            [(0, 1), (0, 2), (1, 2), (2, 0)], n_nodes=4
        )
        result = sonet_motif_statistics(W)
        p2 = result["connection_probability"] ** 2
        np.testing.assert_allclose(
            result["alpha"], result["frequencies"] / p2 - 1.0
        )

    def test_signed_thresholding_and_loop_removal(self) -> None:
        W = np.array(
            [
                [9.0, 0.0, -2.0],
                [0.8, 8.0, 0.0],
                [0.0, 0.4, 7.0],
            ]
        )
        absolute = sonet_motif_statistics(
            W, threshold=0.5, edge_presence="nonzero"
        )
        positive = sonet_motif_statistics(
            W, threshold=0.5, edge_presence="positive"
        )
        self.assertEqual(absolute["n_edges"], 2)
        self.assertEqual(positive["n_edges"], 1)
        self.assertTrue(absolute["self_loops_removed"])

    def test_too_few_nodes_marks_unavailable_frequencies_nan(self) -> None:
        W = package_adjacency([(0, 1)], n_nodes=2)
        result = sonet_motif_statistics(W)
        self.assertTrue(np.all(np.isnan(result["frequencies"][:3])))
        self.assertTrue(np.isfinite(result["frequencies"][3]))


if __name__ == "__main__":
    unittest.main()
