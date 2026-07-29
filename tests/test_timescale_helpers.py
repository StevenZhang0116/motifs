"""Tests for additional timescale helper interfaces."""

from __future__ import annotations

import unittest

import numpy as np

from motif_cumulants import (
    motif_cutoff_times_from_cumulants,
    network_cutoff_time,
    structural_timescale_curve,
)


class TimescaleHelperTests(unittest.TestCase):
    def test_network_cutoff_time_general_order_for_zero_network(self) -> None:
        W = np.zeros((4, 4), dtype=float)

        first = network_cutoff_time(
            W,
            tau_node=0.025,
            asymptotic_order=1.0,
        )
        second = network_cutoff_time(
            W,
            tau_node=0.025,
            asymptotic_order=2.0,
        )

        self.assertAlmostEqual(first["cutoff_time"], 0.025)
        self.assertAlmostEqual(second["cutoff_time"], 0.025)
        self.assertAlmostEqual(first["dc_gain"], 0.025)
        self.assertAlmostEqual(second["dc_gain"], 0.025**2)
        self.assertTrue(first["cutoff_defined"])
        self.assertTrue(second["cutoff_defined"])

    def test_precomputed_cumulant_sequence(self) -> None:
        result = motif_cutoff_times_from_cumulants(
            [0.1, 0.02],
            n_nodes=4,
            tau_node=0.5,
            coupling=0.2,
        )

        expected_terms = np.array([0.04, 0.0032])
        expected_denominators = 1.0 - np.cumsum(expected_terms)
        expected_times = 0.5 / expected_denominators

        np.testing.assert_allclose(result["contributions"], expected_terms)
        np.testing.assert_allclose(
            result["denominators"],
            expected_denominators,
        )
        np.testing.assert_allclose(result["time_constants"], expected_times)
        np.testing.assert_array_equal(result["valid"], [True, True])

    def test_structural_curve_for_regular_cycle(self) -> None:
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
            result["effective_coupling"],
            eta,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result["cutoff_ratio"],
            1.0 / (1.0 - eta),
            atol=1e-14,
        )

    def test_structural_curve_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            structural_timescale_curve(-np.eye(2), eta=[0.5])
        with self.assertRaises(ValueError):
            structural_timescale_curve(np.eye(2), eta=[1.0])
        with self.assertRaises(ValueError):
            structural_timescale_curve(np.zeros((2, 2)), eta=[0.5])

    def test_zero_overlap_nan_policy(self) -> None:
        result = network_cutoff_time(
            np.zeros((2, 2)),
            tau_node=1.0,
            B=np.array([1.0, 0.0]),
            C=np.array([0.0, 1.0]),
            zero_overlap="nan",
        )
        self.assertFalse(result["cutoff_defined"])
        self.assertTrue(np.isnan(result["cutoff_time"]))


if __name__ == "__main__":
    unittest.main()
