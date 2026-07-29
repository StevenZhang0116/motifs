"""Example usage of motif cumulants and network timescales."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow this example to run directly from an unpacked source directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motif_cumulants import (
    chain_motif_cumulants,
    cycle_motif_cumulants,
    exponential_impulse_response,
    exponential_network_timescales,
    motif_cutoff_times_by_order,
    paper_cutoff_time_constant,
)


def print_motif_result(name: str, W: np.ndarray, max_order: int = 6) -> None:
    """Compute and print chain and cycle statistics for one network."""
    chain = chain_motif_cumulants(W, max_order=max_order)
    cycle = cycle_motif_cumulants(W, max_order=max_order)

    print(f"\n{name}")
    print(
        "order    chain moment  chain cumulant  "
        "cycle moment  cycle cumulant"
    )
    print(
        "-----  --------------  --------------  "
        "------------  --------------"
    )
    for order, chain_moment, chain_cumulant, cycle_moment, cycle_cumulant in zip(
        chain["order"],
        chain["moments"],
        chain["cumulants"],
        cycle["moments"],
        cycle["cumulants"],
    ):
        print(
            f"{order:5d}  "
            f"{chain_moment:14.8g}  "
            f"{chain_cumulant:14.8g}  "
            f"{cycle_moment:12.8g}  "
            f"{cycle_cumulant:14.8g}"
        )


def print_timescale_result(W: np.ndarray) -> None:
    """Show exact, motif-truncated, pole, and impulse-response calculations."""
    tau_node = 1.0
    coupling = 0.4

    exact_time = paper_cutoff_time_constant(
        W,
        tau_node=tau_node,
        coupling=coupling,
    )
    motif_time = motif_cutoff_times_by_order(
        W,
        max_order=6,
        tau_node=tau_node,
        coupling=coupling,
    )
    exponential = exponential_network_timescales(
        W,
        tau_node=tau_node,
        coupling=coupling,
        return_poles=True,
    )

    print("\nTime constants for exponential single-node dynamics")
    print(f"node time constant:       {tau_node:.8g}")
    print(f"full-W cutoff time:       {exact_time:.8g}")
    print(f"exponential cutoff time:  {exponential['cutoff_time']:.8g}")
    print(
        "dominant-pole time:     "
        f"{exponential['dominant_pole_time']:.8g}"
    )

    print("\norder  motif cutoff time  relative error  paper-valid")
    print("-----  -----------------  --------------  -----------")
    for order, cutoff, error, valid in zip(
        motif_time["order"],
        motif_time["time_constants"],
        motif_time["relative_error"],
        motif_time["paper_valid"],
    ):
        print(
            f"{order:5d}  {cutoff:17.8g}  "
            f"{error:14.8g}  {str(bool(valid)):>11s}"
        )

    try:
        times = np.linspace(0.0, 5.0, 6)
        response = exponential_impulse_response(
            W,
            times=times,
            tau_node=tau_node,
            coupling=coupling,
        )
    except ImportError:
        print("\nSciPy is not installed; impulse-response example skipped.")
    else:
        print("\nImpulse response")
        for time, value in zip(times, response):
            print(f"t={time:.2f}: {value:.8g}")


def main() -> None:
    # Directed three-node cycle: 1 -> 2 -> 3 -> 1.
    # The convention is W[i, j] = weight of j -> i.
    regular_cycle = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    # An irregular weighted directed network with nonzero higher cumulants.
    weighted_network = np.array(
        [
            [0.0, 0.3, 0.0, 1.1],
            [0.8, 0.0, 0.0, 0.0],
            [0.2, 1.4, 0.0, 0.5],
            [0.0, 0.7, 0.9, 0.0],
        ]
    )

    print_motif_result("Regular directed cycle", regular_cycle)
    print_motif_result("Irregular weighted network", weighted_network)
    print_timescale_result(regular_cycle)


if __name__ == "__main__":
    main()
