"""Tests for weighted/input-output aliases and nonuniform branch weights."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    convergent_motif_cumulants,
    divergent_motif_cumulants,
    generalized_chain_motif_cumulants,
    input_output_chain_motif_cumulants,
    weighted_chain_motif_cumulants,
)


class WeightedAliasTests(unittest.TestCase):
    def test_input_output_aliases_reuse_generalized_implementation(self) -> None:
        W = np.array(
            [[0.0, 0.2, 0.4], [0.8, 0.0, 0.1], [0.3, 0.7, 0.0]]
        )
        B = np.array([1.0, 0.4, 0.2])
        C = np.array([0.3, 0.5, 0.9])
        canonical = generalized_chain_motif_cumulants(W, 5, B=B, C=C)
        input_output = input_output_chain_motif_cumulants(W, 5, B=B, C=C)
        weighted = weighted_chain_motif_cumulants(W, 5, B=B, C=C)
        for result in (input_output, weighted):
            np.testing.assert_allclose(
                result["moments"], canonical["moments"]
            )
            np.testing.assert_allclose(
                result["cumulants"], canonical["cumulants"]
            )

    def test_nonuniform_branch_weights_match_explicit_projectors(self) -> None:
        rng = np.random.default_rng(94)
        W = rng.uniform(-0.1, 0.9, size=(4, 4))
        weights = np.array([0.1, 0.5, 0.7, 0.3])
        u = weights / np.linalg.norm(weights)
        theta = np.eye(4) - np.outer(u, u)
        max_order = 3
        divergent = divergent_motif_cumulants(
            W, max_order, weights=weights
        )
        convergent = convergent_motif_cumulants(
            W, max_order, weights=weights
        )

        bases = [
            np.linalg.matrix_power(W @ theta, order - 1) @ W
            for order in range(1, max_order + 1)
        ]
        for n_index, Bn in enumerate(bases):
            for m_index, Bm in enumerate(bases):
                denominator = W.shape[0] ** (n_index + m_index + 2)
                expected_div = u @ Bn @ theta @ Bm.T @ u / denominator
                expected_conv = u @ Bn.T @ theta @ Bm @ u / denominator
                self.assertAlmostEqual(
                    divergent["cumulants"][n_index, m_index], expected_div
                )
                self.assertAlmostEqual(
                    convergent["cumulants"][n_index, m_index], expected_conv
                )


if __name__ == "__main__":
    unittest.main()
