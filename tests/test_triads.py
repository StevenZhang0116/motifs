"""Tests for the exact induced directed-triad census."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    TRIAD_NAMES,
    classify_directed_triad,
    directed_triad_census,
)


REPRESENTATIVE_EDGES = {
    "003": [],
    "012": [(0, 1)],
    "102": [(0, 1), (1, 0)],
    "021D": [(1, 0), (1, 2)],
    "021U": [(0, 1), (2, 1)],
    "021C": [(0, 1), (1, 2)],
    "111D": [(0, 2), (1, 2), (2, 0)],
    "111U": [(0, 2), (2, 0), (2, 1)],
    "030T": [(0, 1), (0, 2), (2, 1)],
    "030C": [(0, 2), (1, 0), (2, 1)],
    "201": [(0, 1), (1, 0), (0, 2), (2, 0)],
    "120D": [(0, 2), (1, 0), (1, 2), (2, 0)],
    "120U": [(0, 1), (0, 2), (2, 0), (2, 1)],
    "120C": [(0, 1), (0, 2), (1, 2), (2, 0)],
    "210": [(0, 1), (0, 2), (1, 2), (2, 0), (2, 1)],
    "300": [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ],
}


def adjacency_from_edges(n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for source, target in edges:
        W[target, source] = 1.0
    return W


class DirectedTriadCensusTests(unittest.TestCase):
    def test_every_standard_triad_class(self) -> None:
        for expected_name, edges in REPRESENTATIVE_EDGES.items():
            with self.subTest(triad=expected_name):
                result = directed_triad_census(adjacency_from_edges(3, edges))
                self.assertEqual(result["total_triples"], 1)
                self.assertEqual(result["count_by_name"][expected_name], 1)
                self.assertEqual(int(result["counts"].sum()), 1)
                self.assertAlmostEqual(
                    result["proportion_by_name"][expected_name], 1.0
                )

    def test_counts_sum_to_number_of_unordered_triples(self) -> None:
        rng = np.random.default_rng(90)
        W = (rng.random((11, 11)) < 0.23).astype(float)
        np.fill_diagonal(W, 0.0)
        result = directed_triad_census(W)

        expected_total = 11 * 10 * 9 // 6
        self.assertEqual(result["total_triples"], expected_total)
        self.assertEqual(int(result["counts"].sum()), expected_total)
        self.assertAlmostEqual(float(result["proportions"].sum()), 1.0)
        self.assertTupleEqual(tuple(result["triad"]), TRIAD_NAMES)

    def test_all_64_labeled_topologies_match_networkx_when_available(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("NetworkX is not installed.")

        edge_positions = [
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 2),
            (2, 1),
        ]
        for code in range(1 << len(edge_positions)):
            edges = [
                edge
                for bit, edge in enumerate(edge_positions)
                if code & (1 << bit)
            ]
            with self.subTest(edge_code=code):
                W = adjacency_from_edges(3, edges)
                graph = nx.DiGraph()
                graph.add_nodes_from(range(3))
                graph.add_edges_from(edges)
                expected = next(
                    name
                    for name, count in nx.triadic_census(graph).items()
                    if count == 1
                )
                self.assertEqual(classify_directed_triad(W), expected)
                self.assertEqual(
                    directed_triad_census(W)["count_by_name"][expected],
                    1,
                )

    def test_matches_networkx_on_random_graph_when_available(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("NetworkX is not installed.")

        rng = np.random.default_rng(177)
        W = (rng.random((8, 8)) < 0.31).astype(float)
        np.fill_diagonal(W, 0.0)
        result = directed_triad_census(W)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(W.shape[0]))
        targets, sources = np.nonzero(W)
        graph.add_edges_from(zip(sources.tolist(), targets.tolist()))
        expected = nx.triadic_census(graph)

        for name in TRIAD_NAMES:
            self.assertEqual(result["count_by_name"][name], expected[name])

    def test_self_loops_are_ignored(self) -> None:
        W = adjacency_from_edges(3, REPRESENTATIVE_EDGES["030C"])
        np.fill_diagonal(W, 10.0)
        result = directed_triad_census(W)
        self.assertEqual(result["count_by_name"]["030C"], 1)
        self.assertEqual(result["n_edges"], 3)

    def test_signed_threshold_rule(self) -> None:
        W = np.zeros((3, 3), dtype=float)
        W[1, 0] = -2.0
        W[2, 1] = 0.8
        W[0, 2] = 0.1

        nonzero = directed_triad_census(
            W, threshold=0.2, edge_presence="nonzero"
        )
        positive = directed_triad_census(
            W, threshold=0.2, edge_presence="positive"
        )
        self.assertEqual(nonzero["count_by_name"]["021C"], 1)
        self.assertEqual(positive["count_by_name"]["012"], 1)

    def test_fewer_than_three_nodes(self) -> None:
        result = directed_triad_census(np.zeros((2, 2)))
        self.assertEqual(result["total_triples"], 0)
        np.testing.assert_array_equal(result["counts"], 0)
        self.assertTrue(np.all(np.isnan(result["proportions"])))


if __name__ == "__main__":
    unittest.main()
