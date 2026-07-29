"""Tests for network time constants and exponential impulse responses."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from motif_cumulants import (
    exponential_impulse_response,
    exponential_network_timescales,
    motif_cutoff_times_by_order,
    motif_cutoff_times_from_cumulants,
    network_cutoff_time,
    paper_cutoff_time_constant,
    structural_timescale_curve,
)


class NetworkTimescaleTests(unittest.TestCase):
    def test_uncoupled_network_has_single_node_time_constant(self) -> None:
        W = np.zeros((3, 3), dtype=float)
        tau_node = 0.2

        exact = paper_cutoff_time_constant(W, tau_node)
        motif = motif_cutoff_times_by_order(W, 6, tau_node)
        exponential = exponential_network_timescales(
            W,
            tau_node,
            return_poles=True,
        )

        self.assertAlmostEqual(exact, tau_node)
        np.testing.assert_allclose(motif["time_constants"], tau_node)
        self.assertAlmostEqual(motif["exact_time_constant"], tau_node)
        self.assertAlmostEqual(exponential["cutoff_time"], tau_node)
        self.assertAlmostEqual(exponential["dominant_pole_time"], tau_node)
        self.assertTrue(exponential["stable"])
        np.testing.assert_allclose(exponential["poles"], -1.0 / tau_node)

    def test_regular_complete_network_is_exact_at_first_order(self) -> None:
        W = np.ones((4, 4), dtype=float)
        tau_node = 0.2
        coupling = 0.5
        expected = tau_node / (1.0 - 4.0 * coupling * tau_node)

        result = motif_cutoff_times_by_order(
            W,
            max_order=7,
            tau_node=tau_node,
            coupling=coupling,
        )

        self.assertAlmostEqual(result["chain_cumulants"][0], 1.0)
        np.testing.assert_allclose(result["chain_cumulants"][1:], 0.0)
        np.testing.assert_allclose(result["time_constants"], expected)
        self.assertAlmostEqual(result["exact_time_constant"], expected)
        np.testing.assert_allclose(result["relative_error"], 0.0, atol=1e-14)

    def test_high_order_motif_approximation_converges_to_full_matrix(self) -> None:
        W = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.2, 0.0, 0.7],
                [0.4, 0.1, 0.0],
            ]
        )
        result = motif_cutoff_times_by_order(
            W,
            max_order=25,
            tau_node=0.5,
            coupling=0.2,
        )

        self.assertTrue(np.all(result["paper_valid"]))
        self.assertLess(result["relative_error"][-1], 1e-12)
        self.assertAlmostEqual(
            result["time_constants"][-1],
            result["exact_time_constant"],
            places=12,
        )

    def test_structured_full_matrix_result_matches_scalar_wrapper(self) -> None:
        W = np.array([[0.0, 1.0], [0.5, 0.0]])
        structured = network_cutoff_time(
            W,
            tau_node=0.4,
            coupling=0.2,
        )
        scalar = paper_cutoff_time_constant(
            W,
            tau_node=0.4,
            coupling=0.2,
        )

        self.assertTrue(structured["cutoff_defined"])
        self.assertAlmostEqual(structured["cutoff_time"], scalar)
        self.assertAlmostEqual(structured["asymptotic_order"], 1.0)

    def test_precomputed_cumulants_match_matrix_based_sequence(self) -> None:
        W = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        matrix_result = motif_cutoff_times_by_order(
            W,
            max_order=6,
            tau_node=1.0,
            coupling=0.4,
            include_exact=False,
        )
        precomputed = motif_cutoff_times_from_cumulants(
            matrix_result["chain_cumulants"],
            n_nodes=3,
            tau_node=1.0,
            coupling=0.4,
        )

        np.testing.assert_allclose(
            precomputed["time_constants"],
            matrix_result["time_constants"],
        )
        np.testing.assert_allclose(
            precomputed["feedback_terms"],
            matrix_result["feedback_terms"],
        )

    def test_structural_timescale_curve_for_regular_cycle(self) -> None:
        W = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        eta = np.array([0.0, 0.2, 0.5, 0.8])
        result = structural_timescale_curve(W, eta)

        self.assertAlmostEqual(result["spectral_radius"], 1.0)
        np.testing.assert_allclose(
            result["cutoff_ratio"],
            1.0 / (1.0 - eta),
            atol=1e-14,
        )

    def test_general_rolloff_order(self) -> None:
        W = np.ones((3, 3), dtype=float)
        tau_node = 0.4
        coupling = 0.2
        g = 2.0
        denominator = 1.0 - 3.0 * coupling * tau_node**g
        expected = tau_node / denominator ** (1.0 / g)

        exact = paper_cutoff_time_constant(
            W,
            tau_node,
            g=g,
            coupling=coupling,
        )
        motif = motif_cutoff_times_by_order(
            W,
            max_order=4,
            tau_node=tau_node,
            g=g,
            coupling=coupling,
        )

        self.assertAlmostEqual(exact, expected)
        np.testing.assert_allclose(motif["time_constants"], expected)

    def test_custom_input_readout_selects_one_mode(self) -> None:
        W = np.diag([0.2, -0.1])
        B = np.array([1.0, 0.0])
        C = np.array([1.0, 0.0])
        tau_node = 2.0
        expected = 1.0 / (1.0 / tau_node - 0.2)

        cutoff = paper_cutoff_time_constant(
            W,
            tau_node,
            B=B,
            C=C,
        )
        result = exponential_network_timescales(
            W,
            tau_node,
            B=B,
            C=C,
        )

        self.assertAlmostEqual(cutoff, expected)
        self.assertAlmostEqual(result["cutoff_time"], expected)
        self.assertAlmostEqual(result["dominant_pole_time"], expected)

    def test_cutoff_is_undefined_when_C_transpose_B_is_zero(self) -> None:
        W = np.array([[0.0, 0.0], [0.2, 0.0]])
        B = np.array([1.0, 0.0])
        C = np.array([0.0, 1.0])

        with self.assertRaises(ValueError):
            paper_cutoff_time_constant(W, 1.0, B=B, C=C)

        result = exponential_network_timescales(
            W,
            1.0,
            B=B,
            C=C,
        )
        self.assertFalse(result["cutoff_defined"])
        self.assertTrue(np.isnan(result["cutoff_time"]))
        self.assertNotEqual(result["dc_gain"], 0.0)

    def test_unstable_exponential_network_raises_by_default(self) -> None:
        W = np.eye(2)

        with self.assertRaises(ValueError):
            exponential_network_timescales(
                W,
                tau_node=1.0,
                coupling=1.1,
            )

        result = exponential_network_timescales(
            W,
            tau_node=1.0,
            coupling=1.1,
            require_stable=False,
        )
        self.assertFalse(result["stable"])
        self.assertFalse(result["cutoff_defined"])
        self.assertTrue(np.isnan(result["cutoff_time"]))
        self.assertTrue(np.isnan(result["dominant_pole_time"]))

        marginal = exponential_network_timescales(
            W,
            tau_node=1.0,
            coupling=1.0,
            require_stable=False,
        )
        self.assertFalse(marginal["stable"])
        self.assertFalse(marginal["cutoff_defined"])
        self.assertTrue(np.isnan(marginal["cutoff_time"]))
        self.assertTrue(np.isinf(marginal["dominant_pole_time"]))

    def test_invalid_motif_denominator_policies(self) -> None:
        W = np.ones((4, 4), dtype=float)

        magnitude = motif_cutoff_times_by_order(
            W,
            max_order=2,
            tau_node=1.0,
            coupling=0.3,
            invalid_denominator="magnitude",
            include_exact=False,
        )
        self.assertFalse(np.any(magnitude["paper_valid"]))
        self.assertTrue(np.all(np.isfinite(magnitude["time_constants"])))

        nan_result = motif_cutoff_times_by_order(
            W,
            max_order=2,
            tau_node=1.0,
            coupling=0.3,
            invalid_denominator="nan",
            include_exact=False,
        )
        self.assertTrue(np.all(np.isnan(nan_result["time_constants"])))

        with self.assertRaises(ValueError):
            motif_cutoff_times_by_order(
                W,
                max_order=2,
                tau_node=1.0,
                coupling=0.3,
                invalid_denominator="raise",
                include_exact=False,
            )

    def test_impulse_response_matches_single_exponential(self) -> None:
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is not installed.")

        W = np.zeros((2, 2), dtype=float)
        times = np.array([0.0, 0.5, 1.0, 2.0])
        tau_node = 2.0
        actual = exponential_impulse_response(W, times, tau_node)
        expected = np.exp(-times / tau_node)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_impulse_response_accepts_irregular_time_grid(self) -> None:
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is not installed.")

        W = np.zeros((1, 1), dtype=float)
        times = np.array([1.7, 0.0, 0.4])
        actual = exponential_impulse_response(W, times, tau_node=0.8)
        expected = np.exp(-times / 0.8)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_sparse_timescales_match_dense(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")

        W = np.array(
            [
                [0.0, 0.2, 0.0, 0.1],
                [1.0, 0.0, 0.4, 0.0],
                [0.3, 0.0, 0.0, 0.8],
                [0.0, 0.6, 0.5, 0.0],
            ]
        )
        sparse_W = sparse.csr_matrix(W)

        dense = exponential_network_timescales(W, 0.5, coupling=0.1)
        sparse_result = exponential_network_timescales(
            sparse_W,
            0.5,
            coupling=0.1,
        )

        self.assertAlmostEqual(dense["cutoff_time"], sparse_result["cutoff_time"])
        self.assertAlmostEqual(
            dense["dominant_pole_time"],
            sparse_result["dominant_pole_time"],
            places=11,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with_poles = exponential_network_timescales(
                sparse_W,
                0.5,
                coupling=0.1,
                return_poles=True,
            )
        self.assertIn("poles", with_poles)
        self.assertTrue(any("converts" in str(item.message) for item in caught))

    def test_structural_timescale_curve_is_scale_invariant(self) -> None:
        W = np.ones((4, 4), dtype=float)
        eta = np.array([0.0, 0.5, 0.8])
        expected = 1.0 / (1.0 - eta)

        result = structural_timescale_curve(W, eta)
        rescaled = structural_timescale_curve(7.5 * W, eta)

        self.assertAlmostEqual(result["spectral_radius"], 4.0)
        np.testing.assert_allclose(
            result["effective_coupling"],
            eta / 4.0,
        )
        np.testing.assert_allclose(result["cutoff_ratio"], expected)
        np.testing.assert_allclose(
            result["cutoff_ratio"],
            rescaled["cutoff_ratio"],
        )

    def test_structural_timescale_curve_rejects_unsupported_inputs(self) -> None:
        with self.assertRaises(ValueError):
            structural_timescale_curve(np.zeros((2, 2)), [0.5])
        with self.assertRaises(ValueError):
            structural_timescale_curve(
                np.array([[0.0, -1.0], [1.0, 0.0]]),
                [0.5],
            )
        with self.assertRaises(ValueError):
            structural_timescale_curve(np.ones((2, 2)), [1.0])

    def test_impulse_response_accepts_repeated_time_points(self) -> None:
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is not installed.")

        W = np.zeros((1, 1), dtype=float)
        times = np.array([0.5, 0.5, 1.0])
        actual = exponential_impulse_response(W, times, tau_node=2.0)
        expected = np.exp(-times / 2.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            paper_cutoff_time_constant(np.eye(2), 0.0)
        with self.assertRaises(ValueError):
            paper_cutoff_time_constant(np.eye(2), 1.0, g=0.0)
        with self.assertRaises(TypeError):
            motif_cutoff_times_by_order(np.eye(2), True, 1.0)
        with self.assertRaises(ValueError):
            motif_cutoff_times_by_order(
                np.eye(2),
                2,
                1.0,
                invalid_denominator="unknown",
            )
        with self.assertRaises(ValueError):
            exponential_impulse_response(np.eye(2), [-0.1, 0.0], 1.0)
        with self.assertRaises(ValueError):
            exponential_network_timescales(
                np.eye(2),
                1.0,
                B=[1.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
