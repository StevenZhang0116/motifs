"""Tests for one-way bipartite directed-triad enrichment."""

from __future__ import annotations

import unittest
from itertools import combinations
from math import comb

import numpy as np

from motif_cumulants import (
    ONE_WAY_BIPARTITE_TRIAD_NAMES,
    TRIAD_NAMES,
    bipartite_triad_enrichment,
    directed_triad_census,
    lift_bipartite_adjacency,
    one_way_bipartite_triplet_ratios,
)


def brute_force_wedge_counts(forward: np.ndarray) -> tuple[int, int]:
    """Count divergent and convergent wedges by explicit enumeration."""
    n_target, n_source = forward.shape
    divergent = 0
    for source in range(n_source):
        for first, second in combinations(range(n_target), 2):
            if forward[first, source] and forward[second, source]:
                divergent += 1
    convergent = 0
    for target in range(n_target):
        for first, second in combinations(range(n_source), 2):
            if forward[target, first] and forward[target, second]:
                convergent += 1
    return divergent, convergent


class LiftBipartiteAdjacencyTests(unittest.TestCase):
    def test_block_structure_and_groups(self) -> None:
        forward = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
        full, groups = lift_bipartite_adjacency(forward)
        n_target, n_source = forward.shape

        self.assertEqual(full.shape, (n_source + n_target,) * 2)
        # Diagonal blocks must be empty: no within-population edges.
        np.testing.assert_array_equal(full[:n_source, :n_source], 0)
        np.testing.assert_array_equal(full[n_source:, n_source:], 0)
        # Reverse block is empty for a one-way network.
        np.testing.assert_array_equal(full[:n_source, n_source:], 0)
        np.testing.assert_array_equal(full[n_source:, :n_source], forward)
        np.testing.assert_array_equal(groups, [0, 0, 1, 1, 1])
        self.assertEqual(int(full.sum()), int(forward.sum()))

    def test_orientation_places_source_to_target_edge(self) -> None:
        # forward[target, source]: source 0 -> target 1 only.
        forward = np.array([[0, 0], [1, 0]], dtype=int)
        full, _ = lift_bipartite_adjacency(forward)
        # Global node 0,1 are sources; 2,3 are targets. Edge 0 -> 3 means
        # full[3, 0] == 1 under the package convention W[target, source].
        self.assertEqual(int(full[3, 0]), 1)
        self.assertEqual(int(full.sum()), 1)

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            lift_bipartite_adjacency(np.ones(3))
        with self.assertRaises(ValueError):
            lift_bipartite_adjacency(np.empty((0, 3)))
        with self.assertRaises(ValueError):
            lift_bipartite_adjacency(np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            lift_bipartite_adjacency(np.array([[np.nan, 0.0], [1.0, 0.0]]))


class OneWayBipartiteStructureTests(unittest.TestCase):
    def test_only_four_triad_classes_are_realizable(self) -> None:
        """Exhaustively confirm the structural claim for small networks."""
        realized = set()
        for n_source, n_target in ((2, 2), (1, 3), (3, 1), (2, 3)):
            n_cells = n_source * n_target
            for code in range(1 << n_cells):
                flat = [(code >> bit) & 1 for bit in range(n_cells)]
                forward = np.asarray(flat, dtype=int).reshape(
                    n_target, n_source
                )
                full, _ = lift_bipartite_adjacency(forward)
                census = directed_triad_census(full)
                realized.update(
                    name
                    for name, count in census["count_by_name"].items()
                    if count > 0
                )
        self.assertEqual(realized, set(ONE_WAY_BIPARTITE_TRIAD_NAMES))
        impossible = set(TRIAD_NAMES) - set(ONE_WAY_BIPARTITE_TRIAD_NAMES)
        self.assertEqual(realized & impossible, set())

    def test_census_counts_match_brute_force_wedges(self) -> None:
        rng = np.random.default_rng(11)
        forward = (rng.random((5, 4)) < 0.4).astype(int)
        full, _ = lift_bipartite_adjacency(forward)
        census = directed_triad_census(full)

        divergent, convergent = brute_force_wedge_counts(forward)
        # 021D is one source projecting to two targets; 021U is two sources
        # converging on one target.
        self.assertEqual(census["count_by_name"]["021D"], divergent)
        self.assertEqual(census["count_by_name"]["021U"], convergent)

    def test_wedge_counts_match_degree_formulas(self) -> None:
        rng = np.random.default_rng(5)
        forward = (rng.random((6, 5)) < 0.5).astype(int)
        out_degree = forward.sum(axis=0)
        in_degree = forward.sum(axis=1)
        divergent, convergent = brute_force_wedge_counts(forward)
        self.assertEqual(divergent, sum(comb(int(d), 2) for d in out_degree))
        self.assertEqual(convergent, sum(comb(int(d), 2) for d in in_degree))


class AnalyticalOneWayRatioTests(unittest.TestCase):
    def test_observed_counts_match_exact_triad_census(self) -> None:
        """The analytic wedge counts must equal the O(N^3) census exactly."""
        rng = np.random.default_rng(21)
        for shape in ((5, 4), (6, 3), (4, 7), (8, 8)):
            for density in (0.2, 0.5, 0.8):
                forward = (rng.random(shape) < density).astype(int)
                analytic = one_way_bipartite_triplet_ratios(forward)
                full, _ = lift_bipartite_adjacency(forward)
                census = directed_triad_census(full)
                with self.subTest(shape=shape, density=density):
                    self.assertEqual(
                        analytic["observed_divergent"],
                        census["count_by_name"]["021D"],
                    )
                    self.assertEqual(
                        analytic["observed_convergent"],
                        census["count_by_name"]["021U"],
                    )

    def test_expected_counts_match_bernoulli_simulation(self) -> None:
        n_target, n_source, p = 6, 5, 0.4
        rng = np.random.default_rng(31)
        divergent = []
        convergent = []
        for _ in range(4000):
            sample = (rng.random((n_target, n_source)) < p).astype(int)
            counts = one_way_bipartite_triplet_ratios(sample)
            divergent.append(counts["observed_divergent"])
            convergent.append(counts["observed_convergent"])

        expected_divergent = n_source * comb(n_target, 2) * p**2
        expected_convergent = n_target * comb(n_source, 2) * p**2
        self.assertAlmostEqual(
            float(np.mean(divergent)), expected_divergent, delta=0.25
        )
        self.assertAlmostEqual(
            float(np.mean(convergent)), expected_convergent, delta=0.25
        )

    def test_hub_source_is_divergently_enriched(self) -> None:
        forward = np.zeros((8, 6), dtype=int)
        forward[:, 0] = 1
        result = one_way_bipartite_triplet_ratios(forward)
        self.assertEqual(result["observed_divergent"], comb(8, 2))
        self.assertEqual(result["observed_convergent"], 0)
        self.assertGreater(result["divergent_ratio"], 1.0)
        self.assertEqual(result["convergent_ratio"], 0.0)

    def test_empty_matrix_yields_nan_ratios(self) -> None:
        result = one_way_bipartite_triplet_ratios(np.zeros((4, 3), dtype=int))
        self.assertEqual(result["edge_probability"], 0.0)
        self.assertTrue(np.isnan(result["divergent_ratio"]))
        self.assertTrue(np.isnan(result["convergent_ratio"]))

    def test_complete_matrix_has_unit_ratios(self) -> None:
        result = one_way_bipartite_triplet_ratios(np.ones((5, 4), dtype=int))
        self.assertEqual(result["edge_probability"], 1.0)
        self.assertAlmostEqual(result["divergent_ratio"], 1.0)
        self.assertAlmostEqual(result["convergent_ratio"], 1.0)

    def test_metadata_and_invalid_input(self) -> None:
        result = one_way_bipartite_triplet_ratios(
            np.array([[1, 0, 1], [0, 0, 1]], dtype=int)
        )
        self.assertEqual(result["n_source"], 3)
        self.assertEqual(result["n_target"], 2)
        self.assertEqual(result["n_edges"], 3)
        with self.assertRaises(ValueError):
            one_way_bipartite_triplet_ratios(np.full((2, 2), 0.5))


class BipartiteTriadEnrichmentTests(unittest.TestCase):
    def test_impossible_classes_are_nan_and_flagged(self) -> None:
        rng = np.random.default_rng(3)
        forward = (rng.random((4, 4)) < 0.4).astype(int)
        result = bipartite_triad_enrichment(
            forward, n_random=20, random_state=0
        )
        names = list(result["triad"])
        ratios = result["relative_occurrence"]
        possible = result["structurally_possible"]

        for name, ratio, flag in zip(names, ratios, possible):
            if name in ONE_WAY_BIPARTITE_TRIAD_NAMES:
                continue
            self.assertTrue(np.isnan(ratio), f"{name} should be NaN")
            self.assertFalse(bool(flag), f"{name} should be impossible")
            self.assertEqual(int(result["observed_counts"][names.index(name)]), 0)

    def test_mixed_counts_exclude_structurally_empty_triples(self) -> None:
        forward = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
        n_target, n_source = forward.shape
        result = bipartite_triad_enrichment(
            forward, n_random=10, random_state=1
        )

        expected_empty = comb(n_source, 3) + comb(n_target, 3)
        expected_mixed = (
            comb(n_source, 2) * n_target + n_source * comb(n_target, 2)
        )
        self.assertEqual(result["structurally_empty_triplets"], expected_empty)
        self.assertEqual(result["n_mixed_triplets"], expected_mixed)

        # Mixed counts must partition the mixed triples exactly.
        self.assertAlmostEqual(
            float(result["mixed_observed_counts"].sum()), expected_mixed
        )
        self.assertAlmostEqual(
            float(result["mixed_null_mean"].sum()), expected_mixed
        )
        self.assertAlmostEqual(
            float(result["mixed_observed_fractions"].sum()), 1.0
        )
        # The full census still sums to every triple in the lifted network.
        self.assertEqual(
            int(result["observed_counts"].sum()),
            comb(n_source + n_target, 3),
        )

    def test_empty_correction_leaves_z_score_unchanged(self) -> None:
        """The subtracted constant must not perturb the inferential fields."""
        from motif_cumulants import triad_enrichment

        rng = np.random.default_rng(9)
        forward = (rng.random((4, 3)) < 0.5).astype(int)
        full, groups = lift_bipartite_adjacency(forward)

        direct = triad_enrichment(
            full,
            n_random=25,
            null_model="block",
            groups=groups,
            random_state=7,
        )
        wrapped = bipartite_triad_enrichment(
            forward, n_random=25, random_state=7
        )
        np.testing.assert_allclose(
            np.nan_to_num(direct["z_score"], nan=0.0),
            np.nan_to_num(wrapped["z_score"], nan=0.0),
        )
        np.testing.assert_allclose(
            direct["empirical_p_two_sided"],
            wrapped["empirical_p_two_sided"],
        )
        np.testing.assert_array_equal(
            direct["observed_counts"], wrapped["observed_counts"]
        )

    def test_null_preserves_edge_count_and_bipartite_structure(self) -> None:
        rng = np.random.default_rng(2)
        forward = (rng.random((5, 4)) < 0.45).astype(int)
        result = bipartite_triad_enrichment(
            forward, n_random=30, random_state=4, return_samples=True
        )
        samples = result["null_samples"]
        self.assertEqual(samples.shape, (30, len(TRIAD_NAMES)))

        impossible = [
            index
            for index, name in enumerate(TRIAD_NAMES)
            if name not in ONE_WAY_BIPARTITE_TRIAD_NAMES
        ]
        # No randomization may create a structurally impossible triad, which
        # would mean the null fabricated within-population edges.
        self.assertEqual(float(samples[:, impossible].sum()), 0.0)

        mixed_samples = result["mixed_null_samples"]
        np.testing.assert_allclose(
            mixed_samples.sum(axis=1), result["n_mixed_triplets"]
        )

    def test_divergent_enrichment_detected_for_hub_source(self) -> None:
        # One source projects to every target; remaining edges are spread out
        # so that the total edge count is matched but wedges are concentrated.
        n_target, n_source = 8, 6
        forward = np.zeros((n_target, n_source), dtype=int)
        forward[:, 0] = 1
        result = bipartite_triad_enrichment(
            forward, n_random=200, random_state=0
        )
        names = list(result["triad"])
        divergent = result["relative_occurrence"][names.index("021D")]
        convergent = result["relative_occurrence"][names.index("021U")]

        # A single hub source maximizes divergent wedges and eliminates
        # convergent ones entirely.
        self.assertGreater(divergent, 1.0)
        self.assertEqual(
            int(result["observed_counts"][names.index("021U")]), 0
        )
        self.assertLess(convergent, 1.0)

    def test_homogeneous_reference_reports_finite_ratios(self) -> None:
        rng = np.random.default_rng(17)
        forward = (rng.random((6, 6)) < 0.5).astype(int)
        result = bipartite_triad_enrichment(
            forward, n_random=100, random_state=0
        )
        names = list(result["triad"])
        for name in ONE_WAY_BIPARTITE_TRIAD_NAMES:
            ratio = result["relative_occurrence"][names.index(name)]
            self.assertTrue(np.isfinite(ratio), f"{name} ratio should be finite")
            self.assertGreater(ratio, 0.0)
        np.testing.assert_allclose(
            result["log2_relative_occurrence"][names.index("012")],
            np.log2(result["relative_occurrence"][names.index("012")]),
        )

    def test_reported_metadata(self) -> None:
        forward = np.array([[1, 1, 0], [0, 1, 0]], dtype=int)
        result = bipartite_triad_enrichment(
            forward, n_random=5, random_state=0
        )
        self.assertEqual(result["n_source"], 3)
        self.assertEqual(result["n_target"], 2)
        self.assertEqual(result["n_edges"], 3)
        self.assertAlmostEqual(result["edge_density"], 3 / 6)
        self.assertEqual(result["null_model"], "block")
        self.assertEqual(result["n_random"], 5)
        self.assertNotIn("mixed_null_samples", result)

    def test_reproducible_across_calls(self) -> None:
        rng = np.random.default_rng(8)
        forward = (rng.random((5, 5)) < 0.4).astype(int)
        first = bipartite_triad_enrichment(
            forward, n_random=15, random_state=42
        )
        second = bipartite_triad_enrichment(
            forward, n_random=15, random_state=42
        )
        np.testing.assert_allclose(
            first["relative_occurrence"],
            second["relative_occurrence"],
            equal_nan=True,
        )

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            bipartite_triad_enrichment(np.ones((1, 1), dtype=int))
        with self.assertRaises(ValueError):
            bipartite_triad_enrichment(np.ones((2, 2), dtype=int), n_random=1)
        with self.assertRaises(ValueError):
            bipartite_triad_enrichment(np.full((2, 2), 2.0))


if __name__ == "__main__":
    unittest.main()
