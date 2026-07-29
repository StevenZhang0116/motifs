"""Validation tests for the explicit null-model and enrichment APIs."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    block_density_matched_null,
    density_matched_null,
    directed_degree_preserving_null,
    directed_triad_census,
    shuffle_edge_weights,
    triad_enrichment,
)


def _weighted_graph() -> np.ndarray:
    """Return a loop-free weighted graph in W[target, source] orientation."""
    W = np.zeros((7, 7), dtype=float)
    edges = [
        (0, 1, 0.2),
        (0, 2, -0.4),
        (1, 2, 0.7),
        (1, 3, 0.9),
        (2, 0, 1.1),
        (2, 4, -1.3),
        (3, 0, 1.5),
        (3, 5, 1.7),
        (4, 1, -1.9),
        (4, 6, 2.1),
        (5, 2, 2.3),
        (5, 6, -2.5),
        (6, 3, 2.7),
        (6, 4, 2.9),
    ]
    for source, target, weight in edges:
        W[target, source] = weight
    return W


def _support(W: np.ndarray) -> np.ndarray:
    result = np.asarray(W != 0.0)
    result = result.copy()
    np.fill_diagonal(result, False)
    return result


def _block_values(
    W: np.ndarray,
    groups: np.ndarray,
    target_group: str,
    source_group: str,
) -> np.ndarray:
    target = np.flatnonzero(groups == target_group)
    source = np.flatnonzero(groups == source_group)
    block = W[np.ix_(target, source)].copy()
    if target_group == source_group:
        # The selected within-group block has matching row and column order.
        np.fill_diagonal(block, 0.0)
    return np.sort(block[block != 0.0])


class ExplicitNullModelTests(unittest.TestCase):
    def test_density_matched_null_preserves_edges_and_weight_multiset(self) -> None:
        W = _weighted_graph()
        first = density_matched_null(
            W,
            random_state=17,
            preserve_weight_distribution=True,
        )
        second = density_matched_null(
            W,
            random_state=17,
            preserve_weight_distribution=True,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(_support(first).sum()), int(_support(W).sum()))
        self.assertFalse(np.any(np.diag(first)))
        np.testing.assert_allclose(
            np.sort(first[first != 0.0]),
            np.sort(W[W != 0.0]),
        )

    def test_degree_preserving_null_preserves_every_binary_degree(self) -> None:
        W = _weighted_graph()
        randomized = directed_degree_preserving_null(
            W,
            n_swaps=12,
            max_tries=20000,
            random_state=9,
            preserve_weight_distribution=True,
            strict=True,
        )

        original_support = _support(W)
        randomized_support = _support(randomized)
        # Row sums are in-degrees and column sums are out-degrees under the
        # package convention W[target, source].
        np.testing.assert_array_equal(
            randomized_support.sum(axis=1), original_support.sum(axis=1)
        )
        np.testing.assert_array_equal(
            randomized_support.sum(axis=0), original_support.sum(axis=0)
        )
        np.testing.assert_allclose(
            np.sort(randomized[randomized != 0.0]),
            np.sort(W[W != 0.0]),
        )
        self.assertFalse(np.any(np.diag(randomized)))

    def test_block_density_null_preserves_each_oriented_block(self) -> None:
        W = _weighted_graph()
        groups = np.array(["E", "E", "E", "I", "I", "M", "M"])
        randomized = block_density_matched_null(
            W,
            groups,
            random_state=22,
            preserve_weight_distribution=True,
        )

        for target_group in ("E", "I", "M"):
            for source_group in ("E", "I", "M"):
                with self.subTest(
                    target_group=target_group,
                    source_group=source_group,
                ):
                    np.testing.assert_allclose(
                        _block_values(
                            randomized,
                            groups,
                            target_group,
                            source_group,
                        ),
                        _block_values(
                            W,
                            groups,
                            target_group,
                            source_group,
                        ),
                    )
        self.assertFalse(np.any(np.diag(randomized)))

    def test_weight_shuffle_preserves_support_and_exact_weights(self) -> None:
        W = _weighted_graph()
        first = shuffle_edge_weights(W, random_state=123)
        second = shuffle_edge_weights(W, random_state=123)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(_support(first), _support(W))
        np.testing.assert_allclose(
            np.sort(first[first != 0.0]),
            np.sort(W[W != 0.0]),
        )

    def test_direct_triad_enrichment_is_reproducible(self) -> None:
        W = _weighted_graph()
        first = triad_enrichment(
            W,
            n_random=12,
            null_model="density",
            random_state=31,
            return_samples=True,
        )
        second = triad_enrichment(
            W,
            n_random=12,
            null_model="density",
            random_state=31,
            return_samples=True,
        )
        census = directed_triad_census(W)

        np.testing.assert_array_equal(
            first["observed_counts"], census["counts"]
        )
        np.testing.assert_array_equal(first["null_samples"], second["null_samples"])
        np.testing.assert_allclose(first["z_score"], second["z_score"], equal_nan=True)
        self.assertEqual(first["null_samples"].shape, (12, 16))
        self.assertTrue(
            np.all(
                (first["empirical_p_two_sided"] > 0.0)
                & (first["empirical_p_two_sided"] <= 1.0)
            )
        )

    def test_block_enrichment_requires_groups(self) -> None:
        with self.assertRaises(ValueError):
            triad_enrichment(
                _weighted_graph(),
                n_random=5,
                null_model="block",
                random_state=0,
            )


if __name__ == "__main__":
    unittest.main()
