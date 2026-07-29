"""Tests for directed null networks and triad enrichment."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    directed_triad_census,
    randomize_directed_adjacency,
    triad_motif_enrichment,
)


def random_binary_graph(seed: int = 3, n_nodes: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = (rng.random((n_nodes, n_nodes)) < 0.32).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


class NullModelTests(unittest.TestCase):
    def test_degree_preserving_swap_keeps_all_degrees(self) -> None:
        W = random_binary_graph()
        randomized = randomize_directed_adjacency(
            W,
            null_model="degree_preserving",
            n_swaps=40,
            max_tries=10000,
            random_state=12,
        )
        np.testing.assert_array_equal(randomized.sum(axis=1), W.sum(axis=1))
        np.testing.assert_array_equal(randomized.sum(axis=0), W.sum(axis=0))
        self.assertEqual(int(randomized.sum()), int(W.sum()))
        self.assertFalse(np.any(np.diag(randomized)))

    def test_fixed_edge_null_preserves_density_exactly(self) -> None:
        W = random_binary_graph(seed=7)
        randomized = randomize_directed_adjacency(
            W, null_model="fixed_edges", random_state=99
        )
        self.assertEqual(int(randomized.sum()), int(W.sum()))
        self.assertFalse(np.any(np.diag(randomized)))

    def test_erdos_renyi_null_is_reproducible_for_a_seed(self) -> None:
        W = random_binary_graph(seed=11)
        first = randomize_directed_adjacency(
            W, null_model="erdos_renyi", random_state=5
        )
        second = randomize_directed_adjacency(
            W, null_model="erdos_renyi", random_state=5
        )
        np.testing.assert_array_equal(first, second)

    def test_enrichment_shapes_ranges_and_observed_counts(self) -> None:
        W = random_binary_graph(seed=20, n_nodes=7)
        result = triad_motif_enrichment(
            W,
            n_random=20,
            null_model="fixed_edges",
            random_state=123,
            return_null_counts=True,
        )
        census = directed_triad_census(W)
        np.testing.assert_array_equal(
            result["observed_counts"], census["counts"]
        )
        self.assertEqual(result["null_counts"].shape, (20, 16))
        self.assertEqual(result["z_scores"].shape, (16,))
        self.assertTrue(
            np.all(
                (result["empirical_p_two_sided"] >= 0.0)
                & (result["empirical_p_two_sided"] <= 1.0)
            )
        )
        profile_norm = np.linalg.norm(result["significance_profile"])
        self.assertTrue(np.isclose(profile_norm, 0.0) or np.isclose(profile_norm, 1.0))

    def test_invalid_null_settings(self) -> None:
        with self.assertRaises(ValueError):
            randomize_directed_adjacency(
                np.eye(3), null_model="unknown"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            triad_motif_enrichment(np.eye(3), n_random=1)


if __name__ == "__main__":
    unittest.main()
