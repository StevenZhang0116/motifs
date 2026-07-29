"""Example: exact and motif-based network time constants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow this example to run directly from an unpacked source directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motif_cumulants import (
    exponential_impulse_response,
    exponential_network_timescales,
    motif_cutoff_times_by_order,
    paper_cutoff_time_constant,
)


def main() -> None:
    # W is a structural weighted adjacency matrix. The actual recurrent
    # connectivity in the dynamical model is coupling * W.
    W = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.2, 0.0, 0.7],
            [0.4, 0.1, 0.0],
        ]
    )

    tau_node = 0.050  # 50 ms
    coupling = 3.0
    max_order = 10

    exact_cutoff = paper_cutoff_time_constant(
        W,
        tau_node,
        coupling=coupling,
    )
    motif = motif_cutoff_times_by_order(
        W,
        max_order=max_order,
        tau_node=tau_node,
        coupling=coupling,
    )
    exponential = exponential_network_timescales(
        W,
        tau_node,
        coupling=coupling,
    )

    print(f"Exact paper-style cutoff time: {exact_cutoff:.6f} s")
    print(
        "Dominant-pole decay time:      "
        f"{exponential['dominant_pole_time']:.6f} s"
    )
    print("\nMotif approximations")
    print("order   time (s)   relative error   paper-valid")
    print("-----  ---------  ---------------  -----------")
    for order, time_constant, error, valid in zip(
        motif["order"],
        motif["time_constants"],
        motif["relative_error"],
        motif["paper_valid"],
    ):
        print(
            f"{order:5d}  {time_constant:9.6f}  "
            f"{error:15.6g}  {str(bool(valid)):>11s}"
        )

    # The impulse response requires SciPy.
    times = np.linspace(0.0, 0.5, 101)
    response = exponential_impulse_response(
        W,
        times,
        tau_node,
        coupling=coupling,
    )
    print("\nFirst five impulse-response values:")
    for time, value in zip(times[:5], response[:5]):
        print(f"t={time:.3f} s: {value:.8f}")


if __name__ == "__main__":
    main()
