"""Tests for arbitrary input/readout chain motif cumulants."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants.chain import chain_motif_cumulants
from motif_cumulants.generalized import (
    generalized_chain_motif_cumulants,
    generalized_chain_motif_moments,
)


class GeneralizedChainTests(unittest.TestCase):
    def test_matches_explicit_matrix_formulas(self) -> None:
        rng = np.random.default_rng(12345)
        W = rng.uniform(-0.4, 0.8, size=(5, 5))
        B = np.array([1.0, -0.2, 0.5, 0.7, 0.1])
        C = np.array([0.4, 0.6, -0.3, 0.2, 0.8])
        max_order = 5
        N = W.shape[0]
        overlap = float(C @ B)
        theta = np.eye(N) - np.outer(B, C) / overlap

        result = generalized_chain_motif_cumulants(
            W, max_order, B=B, C=C
        )
        expected_moments = np.empty(max_order)
        expected_cumulants = np.empty(max_order)
        for index in range(max_order):
            order = index + 1
            expected_moments[index] = (
                C @ np.linalg.matrix_power(W, order) @ B
                / (N**order * overlap)
            )
            expected_cumulants[index] = (
                C
                @ W
                @ np.linalg.matrix_power(theta @ W, order - 1)
                @ B
                / (N**order * overlap)
            )

        np.testing.assert_allclose(result["moments"], expected_moments)
        np.testing.assert_allclose(result["cumulants"], expected_cumulants)

    def test_uniform_vectors_recover_standard_chain(self) -> None:
        rng = np.random.default_rng(73)
        W = rng.uniform(size=(7, 7))
        u = np.ones(7) / np.sqrt(7)
        standard = chain_motif_cumulants(W, 6)
        generalized = generalized_chain_motif_cumulants(
            W, 6, B=u, C=u
        )
        np.testing.assert_allclose(
            generalized["moments"], standard["moments"], atol=1e-14
        )
        np.testing.assert_allclose(
            generalized["cumulants"], standard["cumulants"], atol=1e-14
        )

    def test_invariant_to_nonzero_vector_rescaling(self) -> None:
        W = np.array(
            [
                [0.0, 0.3, 0.7],
                [0.8, 0.0, 0.2],
                [0.1, 0.6, 0.0],
            ]
        )
        B = np.array([1.0, 0.4, -0.2])
        C = np.array([0.5, -0.1, 0.9])
        base = generalized_chain_motif_cumulants(W, 5, B=B, C=C)
        scaled = generalized_chain_motif_cumulants(
            W, 5, B=-3.0 * B, C=2.5 * C
        )
        np.testing.assert_allclose(base["moments"], scaled["moments"])
        np.testing.assert_allclose(base["cumulants"], scaled["cumulants"])

    def test_moment_only_function(self) -> None:
        W = np.array([[0.0, 1.0], [0.5, 0.0]])
        B = np.array([1.0, 0.2])
        C = np.array([0.4, 1.0])
        expected = generalized_chain_motif_cumulants(
            W, 4, B=B, C=C
        )["moments"]
        actual = generalized_chain_motif_moments(W, 4, B=B, C=C)
        np.testing.assert_allclose(actual, expected)

    def test_sparse_matches_dense(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")
        W = np.array(
            [
                [0.0, 0.3, 0.0],
                [0.8, 0.0, 0.2],
                [0.1, 0.0, 0.0],
            ]
        )
        B = np.array([1.0, 0.4, 0.2])
        C = np.array([0.5, 0.1, 0.9])
        dense = generalized_chain_motif_cumulants(W, 5, B=B, C=C)
        sparse_result = generalized_chain_motif_cumulants(
            sparse.csr_matrix(W), 5, B=B, C=C
        )
        np.testing.assert_allclose(dense["moments"], sparse_result["moments"])
        np.testing.assert_allclose(
            dense["cumulants"], sparse_result["cumulants"]
        )

    def test_invalid_overlap_and_shapes(self) -> None:
        with self.assertRaises(ValueError):
            generalized_chain_motif_cumulants(
                np.eye(3), 2, B=[1, 0, 0], C=[0, 1, 0]
            )
        with self.assertRaises(ValueError):
            generalized_chain_motif_cumulants(
                np.eye(3), 2, B=[1, 0], C=[1, 1, 1]
            )
        with self.assertRaises(TypeError):
            generalized_chain_motif_cumulants(
                np.eye(3), 2, B=[1, 1, 1], C=[1, 1, 1], return_moments=1
            )


if __name__ == "__main__":
    unittest.main()
