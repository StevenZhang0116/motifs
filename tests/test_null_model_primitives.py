"""Invariant tests for the lower-level directed null-network generators."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    block_density_matched_null,
    density_matched_null,
    directed_degree_preserving_null,
    shuffle_edge_weights,
    triad_enrichment,
)


def example_weighted_graph() -> np.ndarray:
    """Return a loop-free weighted graph in W[target, source] convention."""
    return np.array(
        [
            [0.0, 0.2, 0.0, 0.0, 0.8, 0.0],
            [0.7, 0.0, 0.4, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.9, 0.0, 0.5],
            [0.0, 0.6, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.8, 0.4, 0.0, 0.7],
            [0.5, 0.0, 0.0, 0.3, 0.6, 0.0],
        ]
    )


def block_edge_counts(W: np.ndarray, groups: np.ndarray) -> np.ndarray:
    labels = list(dict.fromkeys(groups.tolist()))
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for target_index, target_label in enumerate(labels):
        target_nodes = np.flatnonzero(groups == target_label)
        for source_index, source_label in enumerate(labels):
            source_nodes = np.flatnonzero(groups == source_label)
            block = W[np.ix_(target_nodes, source_nodes)] != 0.0
            if target_label == source_label:
                block = block.copy()
                np.fill_diagonal(block, False)
            counts[target_index, source_index] = int(block.sum())
    return counts


class NullModelPrimitiveTests(unittest.TestCase):
    def test_density_null_preserves_exact_edge_count_and_is_reproducible(self) -> None:
        W = example_weighted_graph()
        first = density_matched_null(W, random_state=17)
        second = density_matched_null(W, random_state=17)

        self.assertEqual(int(np.count_nonzero(first)), int(np.count_nonzero(W)))
        self.assertFalse(np.any(np.diag(first)))
        np.testing.assert_array_equal(first, second)

    def test_degree_null_preserves_every_binary_in_and_out_degree(self) -> None:
        W = example_weighted_graph()
        topology = W != 0.0
        randomized = directed_degree_preserving_null(
            W,
            n_swaps=30,
            max_tries=10000,
            random_state=8,
            strict=True,
        )
        randomized_topology = randomized != 0.0

        np.testing.assert_array_equal(
            randomized_topology.sum(axis=1), topology.sum(axis=1)
        )
        np.testing.assert_array_equal(
            randomized_topology.sum(axis=0), topology.sum(axis=0)
        )
        self.assertFalse(np.any(np.diag(randomized_topology)))

    def test_block_null_preserves_every_target_source_block_count(self) -> None:
        W = example_weighted_graph()
        groups = np.array(["E", "E", "E", "I", "I", "I"])
        randomized = block_density_matched_null(
            W,
            groups,
            random_state=22,
        )

        np.testing.assert_array_equal(
            block_edge_counts(randomized, groups),
            block_edge_counts(W, groups),
        )
        self.assertFalse(np.any(np.diag(randomized)))

    def test_weight_shuffle_preserves_support_and_weight_multiset(self) -> None:
        W = example_weighted_graph()
        shuffled = shuffle_edge_weights(W, random_state=31)

        np.testing.assert_array_equal(shuffled != 0.0, W != 0.0)
        np.testing.assert_allclose(
            np.sort(shuffled[shuffled != 0.0]),
            np.sort(W[W != 0.0]),
        )

    def test_modern_enrichment_interface_returns_valid_samples(self) -> None:
        W = example_weighted_graph()
        result = triad_enrichment(
            W,
            n_random=12,
            null_model="density",
            random_state=4,
            return_samples=True,
        )

        self.assertEqual(result["null_samples"].shape, (12, 16))
        self.assertEqual(result["observed_counts"].shape, (16,))
        self.assertEqual(result["z_score"].shape, (16,))
        self.assertTrue(
            np.all(
                (result["empirical_p_two_sided"] > 0.0)
                & (result["empirical_p_two_sided"] <= 1.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
