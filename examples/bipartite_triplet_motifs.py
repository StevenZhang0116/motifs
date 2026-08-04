"""Compare two one-way rectangular networks against block-random baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow this example to run directly from an unpacked source directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motif_cumulants import (
    bipartite_triad_enrichment,
    one_way_bipartite_triplet_ratios,
)


# The triad census underlying the sampled null is O((n_source + n_target)^3)
# per randomization, so keep the demonstration small.
N_SOURCE = 10
N_TARGET = 10
N_RANDOM = 120

rng = np.random.default_rng(0)

# Network 1: homogeneous source-to-target connectivity.
homogeneous = (rng.random((N_TARGET, N_SOURCE)) < 0.25).astype(int)

# Network 2: the same expected density, but a few hub sources carry most of
# the projections, which concentrates divergent wedges.
hub_probability = np.full(N_SOURCE, 0.06)
hub_probability[:3] = 0.85
clustered = (
    rng.random((N_TARGET, N_SOURCE)) < hub_probability[None, :]
).astype(int)

result_homogeneous = bipartite_triad_enrichment(
    homogeneous,
    n_random=N_RANDOM,
    random_state=1,
)
result_clustered = bipartite_triad_enrichment(
    clustered,
    n_random=N_RANDOM,
    random_state=2,
)

print(
    f"homogeneous: {result_homogeneous['n_edges']} edges, "
    f"density {result_homogeneous['edge_density']:.3f}"
)
print(
    f"clustered:   {result_clustered['n_edges']} edges, "
    f"density {result_clustered['edge_density']:.3f}"
)
print(
    f"\nmixed triples per network: "
    f"{result_homogeneous['n_mixed_triplets']}\n"
)

header = (
    f"{'motif':>5s}  {'homog ratio':>11s}  {'homog z':>9s}  "
    f"{'clust ratio':>11s}  {'clust z':>9s}"
)
print(header)
print("-" * len(header))

for name, ratio_1, z_1, ratio_2, z_2 in zip(
    result_homogeneous["triad"],
    result_homogeneous["relative_occurrence"],
    result_homogeneous["z_score"],
    result_clustered["relative_occurrence"],
    result_clustered["z_score"],
):
    # Skip classes that a one-way bipartite network cannot realize.
    if not (np.isfinite(ratio_1) or np.isfinite(ratio_2)):
        continue
    print(
        f"{name:>5s}  {ratio_1:11.4f}  {z_1:9.3f}  "
        f"{ratio_2:11.4f}  {z_2:9.3f}"
    )

print(
    "\nRatios above one indicate enrichment relative to a block-randomized\n"
    "network with the same population sizes and edge count. Each network is\n"
    "compared with its own baseline, so this does not test whether the two\n"
    "networks differ significantly from each other."
)

# The analytical Bernoulli comparison needs no sampling. Its wedge counts are
# exact, so they agree with the census above; only the null differs.
print("\nAnalytical Bernoulli baseline (no randomization required):")
for label, forward in (
    ("homogeneous", homogeneous),
    ("clustered", clustered),
):
    analytic = one_way_bipartite_triplet_ratios(forward)
    print(
        f"  {label:>11s}: "
        f"021D {analytic['observed_divergent']:4d} obs / "
        f"{analytic['expected_divergent']:7.2f} exp = "
        f"{analytic['divergent_ratio']:6.3f}   "
        f"021U {analytic['observed_convergent']:4d} obs / "
        f"{analytic['expected_convergent']:7.2f} exp = "
        f"{analytic['convergent_ratio']:6.3f}"
    )
