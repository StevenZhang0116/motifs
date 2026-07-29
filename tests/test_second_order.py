"""Tests for the standard second-order motif profile."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    chain_motif_cumulants,
    cycle_motif_cumulants,
    second_order_motif_statistics,
)


class SecondOrderMotifTests(unittest.TestCase):
    def test_profile_matches_explicit_formulas(self) -> None:
        W = np.array(
            [
                [0.0, 0.4, 0.2, 0.0],
                [0.8, 0.0, 0.0, 0.1],
                [0.3, 0.7, 0.0, 0.5],
                [0.0, 0.2, 0.9, 0.0],
            ]
        )
        result = second_order_motif_statistics(W)
        n_nodes = W.shape[0]
        u = np.ones(n_nodes) / np.sqrt(n_nodes)
        A = W / n_nodes
        p = float(u @ A @ u)

        expected = np.array(
            [
                u @ A @ A @ u,
                u @ A @ A.T @ u,
                u @ A.T @ A @ u,
                np.trace(A @ A),
            ]
        )

        self.assertAlmostEqual(result["connection"], p)
        np.testing.assert_allclose(result["moments"], expected, atol=1e-14)
        np.testing.assert_allclose(
            result["cumulants"], expected - p**2, atol=1e-14
        )

    def test_matches_existing_chain_and_cycle_functions(self) -> None:
        rng = np.random.default_rng(2026)
        W = rng.uniform(-0.1, 0.8, size=(7, 7))

        profile = second_order_motif_statistics(W)
        chain = chain_motif_cumulants(W, 2)
        cycle = cycle_motif_cumulants(W, 2, method="projector")

        self.assertAlmostEqual(profile["moments"][0], chain["moments"][1])
        self.assertAlmostEqual(profile["cumulants"][0], chain["cumulants"][1])
        self.assertAlmostEqual(profile["moments"][3], cycle["moments"][1])
        self.assertAlmostEqual(
            profile["cycle_order_2_cumulant"], cycle["cumulants"][1]
        )

    def test_complete_graph_has_zero_sonet_excess(self) -> None:
        result = second_order_motif_statistics(np.ones((5, 5)))
        self.assertAlmostEqual(result["connection"], 1.0)
        np.testing.assert_allclose(result["moments"], 1.0, atol=1e-15)
        np.testing.assert_allclose(result["cumulants"], 0.0, atol=1e-15)
        self.assertAlmostEqual(result["cycle_order_2_cumulant"], 0.0)

    def test_remove_self_loops_does_not_mutate_input(self) -> None:
        W = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.0, 1.0, 3.0],
            ]
        )
        original = W.copy()
        retained = second_order_motif_statistics(W)
        removed = second_order_motif_statistics(W, remove_self_loops=True)

        np.testing.assert_array_equal(W, original)
        self.assertTrue(removed["self_loops_removed"])
        self.assertFalse(retained["self_loops_removed"])
        self.assertNotAlmostEqual(
            retained["connection"], removed["connection"]
        )

    def test_sparse_matches_dense(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")

        W = np.array(
            [
                [0.0, 0.4, 0.0],
                [0.8, 0.0, 0.1],
                [0.2, 0.7, 0.0],
            ]
        )
        dense = second_order_motif_statistics(W)
        sparse_result = second_order_motif_statistics(sparse.csr_matrix(W))

        self.assertAlmostEqual(dense["connection"], sparse_result["connection"])
        np.testing.assert_allclose(dense["moments"], sparse_result["moments"])
        np.testing.assert_allclose(
            dense["cumulants"], sparse_result["cumulants"]
        )
        self.assertAlmostEqual(
            dense["cycle_order_2_cumulant"],
            sparse_result["cycle_order_2_cumulant"],
        )

    def test_invalid_boolean(self) -> None:
        with self.assertRaises(TypeError):
            second_order_motif_statistics(
                np.eye(3), remove_self_loops=1  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
