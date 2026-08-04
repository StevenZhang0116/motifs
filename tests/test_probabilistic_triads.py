"""Tests for Udvary/Song probability-based triplet motif ratios."""

from __future__ import annotations

import unittest
from itertools import combinations

import numpy as np

from motif_cumulants import (
    TRIAD_NAMES,
    UDVARY_TRIPLET_MOTIF_NAMES,
    classify_directed_triad,
    triplet_motif_class_probabilities,
    triplet_motif_probability_ratios,
    triplet_motif_probability_ratios_from_edge_probabilities,
    udvary_triplet_motif_probability_ratios,
)
from motif_cumulants.probabilistic_triads import _unrank_unordered_triplets


EDGE_POSITIONS = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
)


def brute_force_class_probabilities(edge_probabilities: np.ndarray) -> dict[str, float]:
    """Independent 64-pattern reference using the public triad classifier."""
    result = {name: 0.0 for name in TRIAD_NAMES}
    for code in range(64):
        outgoing = np.zeros((3, 3), dtype=float)
        probability = 1.0
        for bit, (source, target) in enumerate(EDGE_POSITIONS):
            present = bool((code >> bit) & 1)
            edge_probability = float(edge_probabilities[bit])
            probability *= edge_probability if present else 1.0 - edge_probability
            if present:
                # Package orientation is W[target, source].
                outgoing[target, source] = 1.0
        result[classify_directed_triad(outgoing)] += probability
    return result


def set_triplet_edge_probabilities(
    P: np.ndarray,
    nodes: tuple[int, int, int],
    probabilities: np.ndarray,
) -> None:
    """Populate P[target, source] for one ordered triplet."""
    for probability, (source_position, target_position) in zip(
        probabilities, EDGE_POSITIONS
    ):
        source = nodes[source_position]
        target = nodes[target_position]
        P[target, source] = probability


class ProbabilisticTripletTests(unittest.TestCase):
    def test_udvary_order_contains_each_nonempty_triad_once(self) -> None:
        expected = (
            "300",
            "210",
            "120U",
            "120D",
            "120C",
            "030T",
            "030C",
            "201",
            "111U",
            "111D",
            "021U",
            "021D",
            "021C",
            "102",
            "012",
        )
        self.assertEqual(UDVARY_TRIPLET_MOTIF_NAMES, expected)
        self.assertEqual(set(UDVARY_TRIPLET_MOTIF_NAMES), set(TRIAD_NAMES) - {"003"})

    def test_representative_patterns_match_udvary_motif_ids(self) -> None:
        # One labeled pattern per motif ID, taken from the mapping in the
        # authors' released structural_model/eval_motifs.py.
        representatives = (
            (1, 1, 1, 1, 1, 1),  # 1: 300
            (1, 1, 1, 1, 1, 0),  # 2: 210
            (1, 1, 1, 0, 1, 0),  # 3: 120U
            (1, 1, 0, 1, 0, 1),  # 4: 120D
            (1, 1, 1, 0, 0, 1),  # 5: 120C
            (1, 0, 1, 0, 1, 0),  # 6: 030T
            (1, 0, 0, 1, 1, 0),  # 7: 030C
            (1, 1, 1, 1, 0, 0),  # 8: 201
            (1, 1, 1, 0, 0, 0),  # 9: 111U
            (1, 1, 0, 1, 0, 0),  # 10: 111D
            (0, 0, 1, 0, 1, 0),  # 11: 021U
            (1, 0, 1, 0, 0, 0),  # 12: 021D
            (1, 0, 0, 0, 1, 0),  # 13: 021C
            (1, 1, 0, 0, 0, 0),  # 14: 102
            (1, 0, 0, 0, 0, 0),  # 15: 012
        )
        for index, bits in enumerate(representatives):
            probabilities = triplet_motif_class_probabilities(bits)
            expected = np.zeros(15)
            expected[index] = 1.0
            np.testing.assert_array_equal(probabilities, expected)

    def test_class_probabilities_match_explicit_64_pattern_sum(self) -> None:
        edges = np.array([0.13, 0.41, 0.72, 0.08, 0.56, 0.24])
        expected = brute_force_class_probabilities(edges)
        observed = triplet_motif_class_probabilities(edges, include_empty=True)
        names = ("003",) + UDVARY_TRIPLET_MOTIF_NAMES
        np.testing.assert_allclose(
            observed,
            np.asarray([expected[name] for name in names]),
            atol=1e-15,
        )
        self.assertAlmostEqual(float(observed.sum()), 1.0, places=14)

    def test_deterministic_feedforward_loop_has_probability_one(self) -> None:
        # 0 -> 1, 1 -> 2, and 0 -> 2 is the transitive triad 030T.
        edges = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        probabilities = triplet_motif_class_probabilities(edges)
        expected = np.zeros(15)
        expected[UDVARY_TRIPLET_MOTIF_NAMES.index("030T")] = 1.0
        np.testing.assert_array_equal(probabilities, expected)

    def test_homogeneous_probability_network_has_unit_ratios(self) -> None:
        P = np.full((5, 5), 0.23)
        np.fill_diagonal(P, 0.0)
        result = triplet_motif_probability_ratios(P)
        np.testing.assert_allclose(
            result["relative_to_independent_edges"], 1.0, atol=2e-14
        )
        np.testing.assert_allclose(
            result["relative_to_independent_dyads"], 1.0, atol=2e-14
        )
        self.assertEqual(result["n_triplets"], 10)
        self.assertEqual(result["sampling"], "all_unordered_triplets")

        # For a homogeneous p, each class probability is its number of labeled
        # edge patterns times p^k (1-p)^(6-k).
        p = 0.23
        pattern_counts = {name: 0 for name in TRIAD_NAMES}
        edge_counts = {name: None for name in TRIAD_NAMES}
        for code in range(64):
            outgoing = np.zeros((3, 3), dtype=float)
            for bit, (source, target) in enumerate(EDGE_POSITIONS):
                if (code >> bit) & 1:
                    outgoing[target, source] = 1.0
            name = classify_directed_triad(outgoing)
            pattern_counts[name] += 1
            edge_counts[name] = bin(code).count("1")
        expected = []
        for name in UDVARY_TRIPLET_MOTIF_NAMES:
            k = edge_counts[name]
            assert k is not None
            expected.append(pattern_counts[name] * p**k * (1.0 - p) ** (6 - k))
        np.testing.assert_allclose(result["model_probability"], expected, atol=2e-15)

    def test_correlated_probability_profiles_enrich_complete_triplets(self) -> None:
        P = np.zeros((6, 6), dtype=float)
        low = np.full(6, 0.1)
        high = np.full(6, 0.9)
        set_triplet_edge_probabilities(P, (0, 1, 2), low)
        set_triplet_edge_probabilities(P, (3, 4, 5), high)
        triplets = np.array([[0, 1, 2], [3, 4, 5]])

        result = triplet_motif_probability_ratios(P, triplets=triplets)
        index = UDVARY_TRIPLET_MOTIF_NAMES.index("300")
        expected_model = 0.5 * (0.1**6 + 0.9**6)
        expected_edge_random = 0.5**6
        expected_pooled_reciprocal = 0.5 * (0.1**2 + 0.9**2)
        expected_dyad_random = expected_pooled_reciprocal**3

        self.assertAlmostEqual(
            result["model_probability"][index], expected_model, places=14
        )
        self.assertAlmostEqual(
            result["independent_edge_probability"][index],
            expected_edge_random,
            places=14,
        )
        self.assertAlmostEqual(
            result["independent_dyad_probability"][index],
            expected_dyad_random,
            places=14,
        )
        self.assertAlmostEqual(
            result["relative_to_independent_edges"][index],
            expected_model / expected_edge_random,
            places=13,
        )
        self.assertAlmostEqual(
            result["doublet_normalized_ratio"][index],
            expected_model / expected_dyad_random,
            places=13,
        )
        self.assertGreater(
            result["relative_to_independent_edges"][index],
            result["doublet_normalized_ratio"][index],
        )

    def test_pre_extracted_edge_rows_match_square_matrix_wrapper(self) -> None:
        P = np.zeros((6, 6), dtype=float)
        first = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        second = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        set_triplet_edge_probabilities(P, (0, 1, 2), first)
        set_triplet_edge_probabilities(P, (3, 4, 5), second)
        matrix_result = triplet_motif_probability_ratios(
            P,
            triplets=np.array([[0, 1, 2], [3, 4, 5]]),
            include_empty=True,
            doublet_baseline="position_specific",
        )
        row_result = triplet_motif_probability_ratios_from_edge_probabilities(
            np.vstack((first, second)),
            include_empty=True,
            doublet_baseline="position_specific",
            chunk_size=1,
        )
        for key in (
            "model_probability",
            "model_probability_standard_error",
            "independent_edge_probability",
            "relative_to_independent_edges",
            "independent_dyad_probability",
            "relative_to_independent_dyads",
            "mean_edge_probability",
            "mean_dyad_state_probability",
        ):
            np.testing.assert_allclose(matrix_result[key], row_result[key])
        self.assertIsNone(row_result["n_nodes"])
        self.assertIsNone(row_result["total_possible_triplets"])
        self.assertEqual(
            row_result["sampling"], "provided_edge_probability_rows"
        )

    def test_explicit_order_controls_the_six_edge_means(self) -> None:
        P = np.zeros((6, 6), dtype=float)
        first = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        second = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        set_triplet_edge_probabilities(P, (0, 1, 2), first)
        set_triplet_edge_probabilities(P, (3, 4, 5), second)
        result = triplet_motif_probability_ratios(
            P,
            triplets=np.array([[0, 1, 2], [3, 4, 5]]),
            doublet_baseline="position_specific",
        )
        np.testing.assert_allclose(
            result["mean_edge_probability"], 0.5 * (first + second)
        )
        self.assertEqual(result["doublet_baseline"], "position_specific")

    def test_position_specific_dyad_baseline_matches_brute_force(self) -> None:
        P = np.zeros((6, 6), dtype=float)
        first = np.array([0.15, 0.55, 0.25, 0.35, 0.75, 0.05])
        second = np.array([0.65, 0.10, 0.45, 0.80, 0.20, 0.60])
        set_triplet_edge_probabilities(P, (0, 1, 2), first)
        set_triplet_edge_probabilities(P, (3, 4, 5), second)
        result = triplet_motif_probability_ratios(
            P,
            triplets=np.array([[0, 1, 2], [3, 4, 5]]),
            include_empty=True,
            doublet_baseline="position_specific",
        )

        rows = []
        for edges in (first, second):
            states = []
            for start in (0, 2, 4):
                forward, reverse = edges[start : start + 2]
                states.append(
                    [
                        (1 - forward) * (1 - reverse),
                        forward * (1 - reverse),
                        (1 - forward) * reverse,
                        forward * reverse,
                    ]
                )
            rows.append(states)
        mean_states = np.mean(rows, axis=0)
        expected = {name: 0.0 for name in TRIAD_NAMES}
        for code in range(64):
            outgoing = np.zeros((3, 3), dtype=float)
            probability = 1.0
            for dyad, first_bit in enumerate((0, 2, 4)):
                forward = (code >> first_bit) & 1
                reverse = (code >> (first_bit + 1)) & 1
                state = forward + 2 * reverse
                probability *= mean_states[dyad, state]
            for bit, (source, target) in enumerate(EDGE_POSITIONS):
                if (code >> bit) & 1:
                    outgoing[target, source] = 1.0
            expected[classify_directed_triad(outgoing)] += probability
        names = ("003",) + UDVARY_TRIPLET_MOTIF_NAMES
        np.testing.assert_allclose(
            result["independent_dyad_probability"],
            [expected[name] for name in names],
            atol=2e-15,
        )

    def test_combination_unranking_matches_lexicographic_enumeration(self) -> None:
        for n_nodes in range(3, 12):
            expected = np.asarray(
                list(combinations(range(n_nodes), 3)), dtype=np.int64
            )
            observed = _unrank_unordered_triplets(
                n_nodes, np.arange(expected.shape[0], dtype=np.int64)
            )
            np.testing.assert_array_equal(observed, expected)

    def test_sampled_triplets_are_reproducible(self) -> None:
        rng = np.random.default_rng(4)
        P = rng.uniform(0.0, 0.4, size=(12, 12))
        np.fill_diagonal(P, 0.0)
        first = triplet_motif_probability_ratios(
            P, sample_size=500, random_state=27, chunk_size=73
        )
        second = triplet_motif_probability_ratios(
            P, sample_size=500, random_state=27, chunk_size=101
        )
        np.testing.assert_allclose(
            first["model_probability"], second["model_probability"], atol=1e-15
        )
        np.testing.assert_allclose(
            first["mean_edge_probability"], second["mean_edge_probability"], atol=1e-15
        )
        self.assertEqual(
            first["sampling"], "uniform_unordered_triplets_with_replacement"
        )

    def test_sparse_probability_matrix_matches_dense(self) -> None:
        try:
            from scipy import sparse
        except ImportError:
            self.skipTest("SciPy is not installed.")
        P = np.array(
            [
                [0.0, 0.2, 0.0, 0.1],
                [0.5, 0.0, 0.4, 0.0],
                [0.3, 0.0, 0.0, 0.6],
                [0.0, 0.7, 0.2, 0.0],
            ]
        )
        dense = triplet_motif_probability_ratios(P, include_empty=True)
        sparse_result = triplet_motif_probability_ratios(
            sparse.csr_matrix(P), include_empty=True
        )
        for key in (
            "model_probability",
            "independent_edge_probability",
            "relative_to_independent_edges",
            "independent_dyad_probability",
            "relative_to_independent_dyads",
            "mean_edge_probability",
            "mean_dyad_state_probability",
        ):
            np.testing.assert_allclose(dense[key], sparse_result[key], atol=1e-15)

    def test_alias_matches_primary_function(self) -> None:
        P = np.full((3, 3), 0.2)
        np.fill_diagonal(P, 0.0)
        primary = triplet_motif_probability_ratios(P)
        alias = udvary_triplet_motif_probability_ratios(P)
        np.testing.assert_allclose(
            primary["model_probability"], alias["model_probability"]
        )

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(np.full((3, 3), 1.1))
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(
                np.zeros((4, 4)), triplets=[[0, 0, 1]]
            )
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(
                np.zeros((4, 4)), triplets=[[0, 1, 2]], sample_size=10
            )
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(
                np.zeros((100, 100)), max_exact_triplets=10
            )
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios(
                np.zeros((4, 4)), doublet_baseline="invalid"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            triplet_motif_class_probabilities(np.ones(5))
        with self.assertRaises(ValueError):
            triplet_motif_class_probabilities(np.full(6, -0.1))
        with self.assertRaises(ValueError):
            triplet_motif_probability_ratios_from_edge_probabilities(
                np.empty((0, 6))
            )


if __name__ == "__main__":
    unittest.main()
