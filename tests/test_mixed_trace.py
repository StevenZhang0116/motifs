"""Tests for mixed trace motif moments and cumulants."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from motif_cumulants import (
    mixed_trace_motif_cumulants,
    mixed_trace_motif_moments,
)


class MixedTraceMotifTests(unittest.TestCase):
    def test_matches_explicit_matrix_formulas(self) -> None:
        rng = np.random.default_rng(861)
        W = rng.uniform(-0.2, 1.0, size=(5, 5))
        max_order = 4
        n_nodes = W.shape[0]
        u = np.ones(n_nodes) / np.sqrt(n_nodes)
        theta = np.eye(n_nodes) - np.outer(u, u)

        result = mixed_trace_motif_cumulants(W, max_order)
        expected_moments = np.empty((max_order, max_order))
        expected_cumulants = np.empty((max_order, max_order))

        powers = [
            np.linalg.matrix_power(W, order)
            for order in range(1, max_order + 1)
        ]
        projected = [
            np.linalg.matrix_power(W @ theta, order - 1) @ W
            for order in range(1, max_order + 1)
        ]

        for n_index in range(max_order):
            n_order = n_index + 1
            for m_index in range(max_order):
                m_order = m_index + 1
                denominator = n_nodes ** (n_order + m_order + 1)
                expected_moments[n_index, m_index] = (
                    np.trace(powers[n_index] @ powers[m_index].T)
                    / denominator
                )
                expected_cumulants[n_index, m_index] = (
                    np.trace(
                        projected[n_index]
                        @ theta
                        @ projected[m_index].T
                        @ theta
                    )
                    / denominator
                )

        np.testing.assert_allclose(
            result["moments"], expected_moments, atol=1e-14
        )
        np.testing.assert_allclose(
            result["cumulants"], expected_cumulants, atol=1e-14
        )

    def test_feedforward_loop_is_detected_by_two_one_moment(self) -> None:
        # 0 -> 1 -> 2 and 0 -> 2.
        W = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        moments = mixed_trace_motif_moments(W, max_order=2)
        self.assertAlmostEqual(moments[1, 0], 1.0 / 3.0**4)
        self.assertAlmostEqual(moments[0, 1], 1.0 / 3.0**4)

    def test_cycle_compatible_normalization_multiplies_by_n(self) -> None:
        rng = np.random.default_rng(11)
        W = rng.uniform(size=(6, 6))

        paper = mixed_trace_motif_cumulants(
            W, 4, normalization="recanatesi"
        )
        compatible = mixed_trace_motif_cumulants(
            W, 4, normalization="cycle_compatible"
        )

        np.testing.assert_allclose(
            compatible["moments"], 6.0 * paper["moments"], atol=1e-14
        )
        np.testing.assert_allclose(
            compatible["cumulants"],
            6.0 * paper["cumulants"],
            atol=1e-14,
        )
        self.assertAlmostEqual(paper["normalization_factor"], 1.0 / 6.0)
        self.assertAlmostEqual(compatible["normalization_factor"], 1.0)

    def test_sparse_moments_match_and_cumulants_warn(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")

        dense = np.array(
            [
                [0.0, 0.4, 0.0, 0.2],
                [0.1, 0.0, 0.5, 0.0],
                [0.7, 0.0, 0.0, 0.3],
                [0.0, 0.6, 0.2, 0.0],
            ]
        )
        sparse_matrix = sparse.csr_matrix(dense)

        np.testing.assert_allclose(
            mixed_trace_motif_moments(dense, 4),
            mixed_trace_motif_moments(sparse_matrix, 4),
            atol=1e-14,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sparse_result = mixed_trace_motif_cumulants(sparse_matrix, 4)
        self.assertTrue(any("dense projector" in str(w.message) for w in caught))
        dense_result = mixed_trace_motif_cumulants(dense, 4)
        np.testing.assert_allclose(
            sparse_result["cumulants"],
            dense_result["cumulants"],
            atol=1e-14,
        )

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            mixed_trace_motif_cumulants(np.eye(3), 0)
        with self.assertRaises(ValueError):
            mixed_trace_motif_cumulants(
                np.eye(3), 2, normalization="other"  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            mixed_trace_motif_cumulants(
                np.eye(3), 2, return_moments=1  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
