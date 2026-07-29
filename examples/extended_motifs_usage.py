"""Example: covariance motifs, SONET motifs, triads, and weighted analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow this example to run directly from an unpacked source directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motif_cumulants import (
    covariance_motif_cumulants,
    cycle_motif_cumulants,
    directed_triad_census,
    generalized_chain_motif_cumulants,
    population_motif_cumulants,
    sonet_motif_statistics,
    triad_enrichment,
)


def main() -> None:
    # Convention: W[target, source] is the weight of source -> target.
    W = np.array(
        [
            [0.0, 0.4, 0.0, 0.2, 0.0, 0.3],
            [0.9, 0.0, 0.3, 0.0, 0.1, 0.0],
            [0.6, 0.5, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.7, 0.1, 0.0, 0.5, 0.0],
            [0.2, 0.0, 0.4, 0.0, 0.0, 0.9],
            [0.0, 0.3, 0.0, 0.6, 0.7, 0.0],
        ]
    )

    covariance = covariance_motif_cumulants(W, max_order=3)
    print("Chain cumulants through order 3")
    print(covariance["chain"]["cumulants"])
    print("\nDivergent cumulants, branch lengths 1..3")
    print(covariance["divergent"]["cumulants"])
    print("\nConvergent cumulants, branch lengths 1..3")
    print(covariance["convergent"]["cumulants"])
    print("\nPLOS mixed-trace cumulants, path lengths 1..3")
    print(covariance["trace"]["cumulants"])

    # PRE closed-walk cycles and PLOS mixed traces are intentionally separate.
    cycle = cycle_motif_cumulants(W, max_order=3)
    print("\nPRE closed-walk cycle cumulants")
    print(cycle["cumulants"])

    sonet = sonet_motif_statistics(W)
    print("\nExact distinct-node SONET profile")
    for name, count, frequency, alpha in zip(
        sonet["motif"],
        sonet["counts"],
        sonet["frequencies"],
        sonet["alpha"],
    ):
        print(
            f"{name:12s} count={count:3d} "
            f"frequency={frequency:8.5f} alpha={alpha:8.5f}"
        )

    triads = directed_triad_census(W)
    print("\nNonzero induced triad classes")
    for name, count in triads["count_by_name"].items():
        if count:
            print(f"{name:4s}: {count}")

    # Classical motif enrichment requires an explicit null ensemble. A small
    # sample count keeps this example fast; real analyses should use more.
    enrichment = triad_enrichment(
        W,
        n_random=20,
        null_model="density",
        random_state=7,
    )
    finite = np.isfinite(enrichment["z_score"])
    if np.any(finite):
        largest = np.flatnonzero(finite)[
            np.argmax(np.abs(enrichment["z_score"][finite]))
        ]
        print(
            "\nLargest absolute triad z-score under density null: "
            f"{enrichment['triad'][largest]} "
            f"({enrichment['z_score'][largest]:.3f})"
        )

    B = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    C = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    selected = generalized_chain_motif_cumulants(
        W,
        max_order=3,
        B=B,
        C=C,
    )
    print("\nInput/readout-specific chain cumulants")
    print(selected["cumulants"])

    groups = np.array(["E", "E", "E", "I", "I", "I"])
    population = population_motif_cumulants(W, groups, max_order=2)
    print("\nPopulation labels")
    print(population["group"])
    print("Population chain cumulants: [order, target group, source group]")
    print(population["cumulants"]["chain"])


if __name__ == "__main__":
    main()
