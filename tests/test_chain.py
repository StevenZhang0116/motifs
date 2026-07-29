"""Tests for chain motif moments and cumulants."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    chain_cumulants_from_moments,
    chain_motif_cumulants,
    chain_motif_moments,
)


class ChainMotifCumulantTests(unittest.TestCase):
    def test_first_four_explicit_cumulant_formulas(self) -> None:
        moments = np.array([0.17, 0.08, 0.051, 0.031])
        cumulants = chain_cumulants_from_moments(moments)

        mu1, mu2, mu3, mu4 = moments
        expected = np.array(
            [
                mu1,
                mu2 - mu1**2,
                mu3 - 2.0 * mu1 * mu2 + mu1**3,
                mu4
                - 2.0 * mu1 * mu3
                - mu2**2
                + 3.0 * mu1**2 * mu2
                - mu1**4,
            ]
        )
        np.testing.assert_allclose(cumulants, expected, atol=1e-14)

    def test_projector_matches_moment_recurrence(self) -> None:
        rng = np.random.default_rng(2026)
        W = rng.uniform(-0.3, 1.2, size=(8, 8))
        np.fill_diagonal(W, 0.0)

        direct = chain_motif_cumulants(
            W,
            max_order=10,
            method="projector",
        )
        recurrent = chain_motif_cumulants(
            W,
            max_order=10,
            method="moments",
        )

        np.testing.assert_allclose(
            direct["moments"],
            recurrent["moments"],
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            direct["cumulants"],
            recurrent["cumulants"],
            rtol=1e-11,
            atol=1e-13,
        )

    def test_regular_directed_cycle_has_only_first_order_cumulant(self) -> None:
        W = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        result = chain_motif_cumulants(W, max_order=8)

        self.assertAlmostEqual(result["cumulants"][0], 1.0 / 3.0)
        np.testing.assert_allclose(
            result["cumulants"][1:],
            0.0,
            atol=1e-15,
        )

    def test_moment_definition_matches_explicit_matrix_powers(self) -> None:
        W = np.array(
            [
                [0.0, 1.0, 0.2],
                [0.5, 0.0, 0.7],
                [1.1, 0.3, 0.0],
            ]
        )
        max_order = 5
        actual = chain_motif_moments(W, max_order)

        n_nodes = W.shape[0]
        expected = np.array(
            [
                np.linalg.matrix_power(W, order).sum()
                / n_nodes ** (order + 1)
                for order in range(1, max_order + 1)
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            chain_motif_cumulants(np.ones((2, 3)), 3)
        with self.assertRaises(ValueError):
            chain_motif_cumulants(np.eye(3), 0)
        with self.assertRaises(TypeError):
            chain_motif_cumulants(np.eye(3), 2.5)
        with self.assertRaises(ValueError):
            chain_motif_cumulants(np.eye(3), 3, method="unknown")
        with self.assertRaises(ValueError):
            chain_cumulants_from_moments([])

    def test_sparse_input_when_scipy_is_available(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")

        dense = np.array(
            [
                [0.0, 0.2, 0.0],
                [1.0, 0.0, 0.4],
                [0.3, 0.0, 0.0],
            ]
        )
        sparse_matrix = sparse.csr_matrix(dense)

        dense_result = chain_motif_cumulants(dense, max_order=7)
        sparse_result = chain_motif_cumulants(sparse_matrix, max_order=7)

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


if __name__ == "__main__":
    unittest.main()
