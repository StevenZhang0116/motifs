"""Tests for the unified covariance-motif interface and terminology."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    chain_motif_cumulants,
    convergent_motif_cumulants,
    covariance_motif_cumulants,
    cycle_motif_moments,
    divergent_motif_cumulants,
    mixed_trace_motif_cumulants,
    mixed_trace_motif_moments,
    trace_motif_cumulants,
    trace_motif_moments,
)


class CovarianceMotifInterfaceTests(unittest.TestCase):
    def test_combined_result_matches_specialized_functions(self) -> None:
        rng = np.random.default_rng(444)
        W = rng.uniform(-0.2, 0.8, size=(5, 5))
        combined = covariance_motif_cumulants(W, max_order=4)

        chain = chain_motif_cumulants(W, 4)
        divergent = divergent_motif_cumulants(W, 4)
        convergent = convergent_motif_cumulants(W, 4)
        trace = mixed_trace_motif_cumulants(W, 4)

        np.testing.assert_allclose(
            combined["chain"]["cumulants"], chain["cumulants"]
        )
        np.testing.assert_allclose(
            combined["divergent"]["cumulants"], divergent["cumulants"]
        )
        np.testing.assert_allclose(
            combined["convergent"]["cumulants"], convergent["cumulants"]
        )
        np.testing.assert_allclose(
            combined["trace"]["cumulants"], trace["cumulants"]
        )

    def test_optional_expensive_sections_can_be_omitted(self) -> None:
        result = covariance_motif_cumulants(
            np.eye(4),
            3,
            include_trace=False,
            include_second_order=False,
            return_moments=False,
        )
        self.assertEqual(
            set(result), {"path_order", "chain", "divergent", "convergent"}
        )
        self.assertNotIn("moments", result["chain"])

    def test_trace_aliases_are_exact_wrappers(self) -> None:
        W = np.array(
            [[0.0, 0.2, 0.8], [0.6, 0.0, 0.1], [0.3, 0.7, 0.0]]
        )
        np.testing.assert_allclose(
            trace_motif_moments(W, 3), mixed_trace_motif_moments(W, 3)
        )
        alias = trace_motif_cumulants(W, 3)
        original = mixed_trace_motif_cumulants(W, 3)
        np.testing.assert_allclose(alias["moments"], original["moments"])
        np.testing.assert_allclose(alias["cumulants"], original["cumulants"])

    def test_pre_cycle_and_plos_trace_are_different_matrix_families(self) -> None:
        W = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.7, 0.0]]
        )
        n_nodes = W.shape[0]
        cycle_order_2 = cycle_motif_moments(W, 2)[1]
        trace_11 = mixed_trace_motif_moments(W, 1)[0, 0]

        self.assertAlmostEqual(cycle_order_2, np.trace(W @ W) / n_nodes**2)
        self.assertAlmostEqual(
            trace_11,
            np.trace(W @ W.T) / n_nodes**3,
        )
        self.assertNotAlmostEqual(cycle_order_2, trace_11)

    def test_invalid_boolean_options(self) -> None:
        with self.assertRaises(TypeError):
            covariance_motif_cumulants(
                np.eye(3), 2, include_trace=1  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
