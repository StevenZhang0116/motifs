"""Tests for population-resolved motif cumulants."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from motif_cumulants.branching import (
    convergent_motif_cumulants,
    divergent_motif_cumulants,
)
from motif_cumulants.chain import chain_motif_cumulants
from motif_cumulants.population import (
    population_branching_motif_cumulants,
    population_chain_motif_cumulants,
    population_motif_cumulants,
)


class PopulationMotifTests(unittest.TestCase):
    def test_single_population_recovers_scalar_cumulants(self) -> None:
        rng = np.random.default_rng(991)
        W = rng.uniform(-0.2, 0.9, size=(6, 6))
        groups = ["all"] * 6
        max_order = 4

        combined = population_motif_cumulants(W, groups, max_order)
        chain = chain_motif_cumulants(W, max_order)
        divergent = divergent_motif_cumulants(W, max_order)
        convergent = convergent_motif_cumulants(W, max_order)

        np.testing.assert_allclose(
            combined["moments"]["chain"][:, 0, 0], chain["moments"]
        )
        np.testing.assert_allclose(
            combined["cumulants"]["chain"][:, 0, 0],
            chain["cumulants"],
        )
        np.testing.assert_allclose(
            combined["moments"]["divergent"][:, :, 0, 0],
            divergent["moments"],
        )
        np.testing.assert_allclose(
            combined["cumulants"]["divergent"][:, :, 0, 0],
            divergent["cumulants"],
        )
        np.testing.assert_allclose(
            combined["moments"]["convergent"][:, :, 0, 0],
            convergent["moments"],
        )
        np.testing.assert_allclose(
            combined["cumulants"]["convergent"][:, :, 0, 0],
            convergent["cumulants"],
        )

    def test_chain_matches_explicit_block_projector_formula(self) -> None:
        W = np.array(
            [
                [0.0, 0.2, 0.4, 0.0],
                [0.8, 0.0, 0.1, 0.3],
                [0.5, 0.0, 0.0, 0.7],
                [0.0, 0.6, 0.9, 0.0],
            ]
        )
        groups = np.array(["E", "E", "I", "I"])
        result = population_chain_motif_cumulants(W, groups, 3)
        H = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        sizes = np.array([2.0, 2.0])
        U = H / np.sqrt(sizes)[None, :]
        theta = np.eye(4) - U @ U.T

        for order in range(1, 4):
            power = np.linalg.matrix_power(W, order)
            basis = np.linalg.matrix_power(W @ theta, order - 1) @ W
            expected_moment = (
                H.T @ power @ H / np.outer(sizes, sizes) / 4 ** (order - 1)
            )
            expected_cumulant = (
                H.T @ basis @ H / np.outer(sizes, sizes) / 4 ** (order - 1)
            )
            np.testing.assert_allclose(
                result["moments"][order - 1], expected_moment
            )
            np.testing.assert_allclose(
                result["cumulants"][order - 1], expected_cumulant
            )

    def test_specialized_and_combined_results_agree(self) -> None:
        rng = np.random.default_rng(8)
        W = rng.uniform(size=(5, 5))
        groups = ["A", "B", "A", "C", "B"]
        combined = population_motif_cumulants(W, groups, 3)
        chain = population_chain_motif_cumulants(W, groups, 3)
        div = population_branching_motif_cumulants(
            W, groups, 3, kind="divergent"
        )
        conv = population_branching_motif_cumulants(
            W, groups, 3, kind="convergent"
        )
        np.testing.assert_array_equal(combined["group"], ["A", "B", "C"])
        np.testing.assert_array_equal(combined["group_size"], [2, 2, 1])
        np.testing.assert_allclose(
            combined["cumulants"]["chain"], chain["cumulants"]
        )
        np.testing.assert_allclose(
            combined["cumulants"]["divergent"], div["cumulants"]
        )
        np.testing.assert_allclose(
            combined["cumulants"]["convergent"], conv["cumulants"]
        )

    def test_sparse_warns_and_matches_dense(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")
        W = np.array(
            [[0.0, 0.2, 0.0], [0.5, 0.0, 0.3], [0.1, 0.4, 0.0]]
        )
        groups = [0, 0, 1]
        dense = population_motif_cumulants(W, groups, 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sparse_result = population_motif_cumulants(
                sparse.csr_matrix(W), groups, 2
            )
        self.assertTrue(any("dense block projector" in str(x.message) for x in caught))
        for name in ("chain", "divergent", "convergent"):
            np.testing.assert_allclose(
                dense["cumulants"][name], sparse_result["cumulants"][name]
            )

    def test_invalid_groups_and_kind(self) -> None:
        with self.assertRaises(ValueError):
            population_chain_motif_cumulants(np.eye(3), [0, 1], 2)
        with self.assertRaises(ValueError):
            population_branching_motif_cumulants(
                np.eye(3), [0, 1, 1], 2, kind="other"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            population_motif_cumulants(
                np.eye(3), [[0], [1], [1]], 2
            )


if __name__ == "__main__":
    unittest.main()
