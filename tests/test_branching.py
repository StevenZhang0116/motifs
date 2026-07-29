"""Tests for convergent and divergent motif moments and cumulants."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    convergent_motif_cumulants,
    convergent_motif_moments,
    divergent_motif_cumulants,
    divergent_motif_moments,
)


class BranchingMotifTests(unittest.TestCase):
    def test_matches_explicit_projector_formulas(self) -> None:
        rng = np.random.default_rng(20260729)
        W = rng.uniform(-0.3, 0.9, size=(6, 6))
        max_order = 4
        n_nodes = W.shape[0]
        u = np.ones(n_nodes) / np.sqrt(n_nodes)
        theta = np.eye(n_nodes) - np.outer(u, u)

        divergent = divergent_motif_cumulants(W, max_order)
        convergent = convergent_motif_cumulants(W, max_order)

        expected_div_moments = np.empty((max_order, max_order))
        expected_conv_moments = np.empty((max_order, max_order))
        expected_div_cumulants = np.empty((max_order, max_order))
        expected_conv_cumulants = np.empty((max_order, max_order))

        moment_powers = [
            np.linalg.matrix_power(W, order)
            for order in range(1, max_order + 1)
        ]
        projected_powers = [
            np.linalg.matrix_power(W @ theta, order - 1) @ W
            for order in range(1, max_order + 1)
        ]

        for n_index in range(max_order):
            n_order = n_index + 1
            for m_index in range(max_order):
                m_order = m_index + 1
                denominator = n_nodes ** (n_order + m_order)
                Wn = moment_powers[n_index]
                Wm = moment_powers[m_index]
                Bn = projected_powers[n_index]
                Bm = projected_powers[m_index]

                expected_div_moments[n_index, m_index] = (
                    u @ Wn @ Wm.T @ u / denominator
                )
                expected_conv_moments[n_index, m_index] = (
                    u @ Wn.T @ Wm @ u / denominator
                )
                expected_div_cumulants[n_index, m_index] = (
                    u @ Bn @ theta @ Bm.T @ u / denominator
                )
                expected_conv_cumulants[n_index, m_index] = (
                    u @ Bn.T @ theta @ Bm @ u / denominator
                )

        np.testing.assert_allclose(
            divergent["moments"], expected_div_moments, atol=1e-14
        )
        np.testing.assert_allclose(
            convergent["moments"], expected_conv_moments, atol=1e-14
        )
        np.testing.assert_allclose(
            divergent["cumulants"], expected_div_cumulants, atol=1e-14
        )
        np.testing.assert_allclose(
            convergent["cumulants"], expected_conv_cumulants, atol=1e-14
        )

    def test_transpose_exchanges_convergent_and_divergent(self) -> None:
        rng = np.random.default_rng(19)
        W = rng.uniform(0.0, 1.0, size=(7, 7))

        convergent = convergent_motif_cumulants(W, 5)
        divergent_transpose = divergent_motif_cumulants(W.T, 5)

        np.testing.assert_allclose(
            convergent["moments"],
            divergent_transpose["moments"],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            convergent["cumulants"],
            divergent_transpose["cumulants"],
            atol=1e-14,
        )

    def test_complete_graph_has_zero_branching_cumulants(self) -> None:
        W = np.ones((5, 5), dtype=float)
        divergent = divergent_motif_cumulants(W, 5)
        convergent = convergent_motif_cumulants(W, 5)

        np.testing.assert_allclose(divergent["moments"], 1.0, atol=1e-15)
        np.testing.assert_allclose(convergent["moments"], 1.0, atol=1e-15)
        np.testing.assert_allclose(divergent["cumulants"], 0.0, atol=1e-15)
        np.testing.assert_allclose(convergent["cumulants"], 0.0, atol=1e-15)

    def test_moment_only_functions_match_result(self) -> None:
        W = np.array(
            [
                [0.0, 0.4, 0.1],
                [0.7, 0.0, 0.2],
                [0.3, 0.9, 0.0],
            ]
        )
        np.testing.assert_allclose(
            divergent_motif_moments(W, 4),
            divergent_motif_cumulants(W, 4)["moments"],
        )
        np.testing.assert_allclose(
            convergent_motif_moments(W, 4),
            convergent_motif_cumulants(W, 4)["moments"],
        )

    def test_sparse_input_matches_dense(self) -> None:
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

        for function in (
            divergent_motif_cumulants,
            convergent_motif_cumulants,
        ):
            dense_result = function(dense, 6)
            sparse_result = function(sparse_matrix, 6)
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

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            divergent_motif_cumulants(np.eye(3), 0)
        with self.assertRaises(ValueError):
            convergent_motif_cumulants(np.ones((2, 3)), 2)
        with self.assertRaises(TypeError):
            divergent_motif_cumulants(
                np.eye(3), 2, return_moments="yes"  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
