"""Tests for cycle motif moments and cumulants."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from motif_cumulants import (
    cycle_cumulants_from_moments,
    cycle_motif_cumulants,
    cycle_motif_moments,
)


class CycleMotifCumulantTests(unittest.TestCase):
    def test_first_four_decomposition_formulas(self) -> None:
        chain = np.array([0.17, -0.025, 0.013, 0.004])
        moments = np.array([0.08, 0.031, -0.012, 0.006])

        actual = cycle_cumulants_from_moments(moments, chain)
        k1, k2, k3, k4 = chain
        expected = np.array(
            [
                moments[0] - k1,
                moments[1] - (2.0 * k2 + k1**2),
                moments[2] - (3.0 * k3 + 3.0 * k1 * k2 + k1**3),
                moments[3]
                - (
                    4.0 * k4
                    + 4.0 * k1 * k3
                    + 2.0 * k2**2
                    + 4.0 * k1**2 * k2
                    + k1**4
                ),
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_cycle_moments_match_explicit_matrix_powers(self) -> None:
        W = np.array(
            [
                [0.1, 0.8, 0.0, 0.2],
                [0.3, 0.0, 0.5, 0.0],
                [0.0, 0.4, -0.2, 0.7],
                [0.6, 0.0, 0.9, 0.0],
            ]
        )
        max_order = 7
        actual = cycle_motif_moments(W, max_order)
        n_nodes = W.shape[0]
        expected = np.array(
            [
                np.trace(np.linalg.matrix_power(W, order))
                / n_nodes**order
                for order in range(1, max_order + 1)
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_projector_matches_explicit_projected_matrix_powers(self) -> None:
        rng = np.random.default_rng(781)
        W = rng.uniform(-0.4, 0.9, size=(6, 6))
        max_order = 8

        actual = cycle_motif_cumulants(
            W,
            max_order,
            method="projector",
        )["cumulants"]

        n_nodes = W.shape[0]
        u = np.ones(n_nodes) / np.sqrt(n_nodes)
        theta = np.eye(n_nodes) - np.outer(u, u)
        projected = theta @ W
        expected = np.array(
            [
                np.trace(np.linalg.matrix_power(projected, order))
                / n_nodes**order
                for order in range(1, max_order + 1)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_projector_matches_moment_decomposition(self) -> None:
        rng = np.random.default_rng(20260728)
        W = rng.uniform(-0.25, 0.75, size=(7, 7))
        np.fill_diagonal(W, rng.uniform(-0.1, 0.2, size=7))

        direct = cycle_motif_cumulants(
            W,
            max_order=9,
            method="projector",
            return_chain_cumulants=True,
        )
        decomposed = cycle_motif_cumulants(
            W,
            max_order=9,
            method="moments",
            return_chain_cumulants=True,
        )

        np.testing.assert_allclose(
            direct["moments"],
            decomposed["moments"],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            direct["chain_cumulants"],
            decomposed["chain_cumulants"],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            direct["cumulants"],
            decomposed["cumulants"],
            rtol=1e-10,
            atol=1e-13,
        )

    def test_complete_graph_has_zero_cycle_cumulants(self) -> None:
        W = np.ones((5, 5), dtype=float)
        result = cycle_motif_cumulants(W, max_order=8)
        np.testing.assert_allclose(result["cumulants"], 0.0, atol=1e-15)
        np.testing.assert_allclose(result["moments"], 1.0, atol=1e-15)

    def test_optional_result_fields(self) -> None:
        result = cycle_motif_cumulants(
            np.eye(3),
            max_order=3,
            return_moments=False,
            return_chain_cumulants=False,
        )
        self.assertEqual(set(result), {"order", "cumulants"})

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            cycle_motif_cumulants(np.ones((2, 3)), 3)
        with self.assertRaises(ValueError):
            cycle_motif_cumulants(np.eye(3), 0)
        with self.assertRaises(TypeError):
            cycle_motif_cumulants(np.eye(3), True)
        with self.assertRaises(ValueError):
            cycle_motif_cumulants(np.eye(3), 3, method="unknown")
        with self.assertRaises(ValueError):
            cycle_cumulants_from_moments([], [0.1])
        with self.assertRaises(ValueError):
            cycle_cumulants_from_moments([0.1, 0.2], [0.1])

    def test_sparse_input_matches_dense_when_scipy_is_available(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")

        dense = np.array(
            [
                [0.0, 0.2, 0.0, 0.1],
                [1.0, 0.0, 0.4, 0.0],
                [0.3, 0.0, 0.0, 0.8],
                [0.0, 0.6, 0.5, 0.0],
            ]
        )
        sparse_matrix = sparse.csr_matrix(dense)

        dense_result = cycle_motif_cumulants(
            dense,
            max_order=7,
            method="moments",
        )
        sparse_result = cycle_motif_cumulants(
            sparse_matrix,
            max_order=7,
            method="auto",
        )

        np.testing.assert_allclose(
            dense_result["moments"],
            sparse_result["moments"],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            dense_result["cumulants"],
            sparse_result["cumulants"],
            atol=1e-14,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            projected_sparse = cycle_motif_cumulants(
                sparse_matrix,
                max_order=5,
                method="projector",
            )
        self.assertTrue(any("converts" in str(item.message) for item in caught))
        projected_dense = cycle_motif_cumulants(
            dense,
            max_order=5,
            method="projector",
        )
        np.testing.assert_allclose(
            projected_sparse["cumulants"],
            projected_dense["cumulants"],
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
