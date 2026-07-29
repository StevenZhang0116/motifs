"""Network time constants and impulse responses from motif cumulants.

This module augments the structural motif calculations with the dynamical
quantities discussed by Hu et al., Physical Review E 98, 062312 (2018):
https://doi.org/10.1103/PhysRevE.98.062312

For uniform input and readout, the paper writes the transfer function as

    G(s) = h(s) / (1 - sum_n N**n * kappa_n * h(s)**n).

If the single-node filter has high-frequency behavior ``h(s) ~ 1 / s**g`` and
single-node cutoff time ``tau_node``, then ``h(0) = tau_node**g``.  The order-K
motif approximation to the network cutoff time is

    tau_K = tau_node /
            |1 - sum_{n=1}^K N**n kappa_n
                 coupling**n tau_node**(n*g)|**(1/g).

The absolute value gives the Bode-magnitude cutoff.  The paper's displayed
positive-response formula applies when the denominator is positive.

Absolute time cannot be inferred from a bare, unscaled adjacency pattern.
One must additionally specify ``tau_node`` and the physical scale multiplying
``W``.  The fully specified exponential-node model used here is

    dx/dt = (coupling * W - I/tau_node) x + B u(t),
    y     = C.T x.

For this model, the scalar impulse response is

    r(t) = C.T exp((coupling*W - I/tau_node)t) B.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Literal, Optional, TypedDict
import warnings

import numpy as np

from ._validation import (
    is_sparse_matrix,
    prepare_adjacency,
    validate_max_order,
    validate_real_vector,
)
from .chain import CumulantMethod, chain_motif_cumulants


InvalidDenominatorPolicy = Literal["magnitude", "nan", "raise"]
UndefinedCutoffPolicy = Literal["nan", "raise"]


class NetworkCutoffResult(TypedDict, total=False):
    """Dictionary returned by :func:`network_cutoff_time`."""

    cutoff_time: float
    cutoff_defined: bool
    dc_gain: float
    instantaneous_gain: float
    normalized_dc_gain: float
    positive_normalized_dc_gain: bool
    asymptotic_order: float


class MotifTimescaleResult(TypedDict, total=False):
    """Dictionary returned by the motif cutoff-time functions."""

    order: np.ndarray
    chain_cumulants: np.ndarray
    feedback_terms: np.ndarray
    contributions: np.ndarray
    cumulative_feedback: np.ndarray
    denominator: np.ndarray
    denominators: np.ndarray
    time_constants: np.ndarray
    cutoff_times: np.ndarray
    paper_valid: np.ndarray
    valid: np.ndarray
    exact_time_constant: float
    full_time_constant: float
    exact_cutoff_time: float
    relative_error: np.ndarray


class ExponentialTimescaleResult(TypedDict, total=False):
    """Dictionary returned by :func:`exponential_network_timescales`."""

    cutoff_time: float
    paper_cutoff_time: float
    cutoff_defined: bool
    dominant_pole_time: float
    dc_gain: float
    dc_response: float
    instantaneous_gain: float
    instantaneous_response: float
    normalized_dc_gain: float
    signed_normalized_impulse_area: float
    positive_normalized_dc_gain: bool
    spectral_abscissa: float
    stability_margin: float
    stable: bool
    poles: np.ndarray


class StructuralTimescaleResult(TypedDict):
    """Dictionary returned by :func:`structural_timescale_curve`."""

    eta: np.ndarray
    spectral_radius: float
    effective_coupling: np.ndarray
    cutoff_ratio: np.ndarray


def _validate_positive_scalar(value: Any, *, name: str) -> float:
    """Validate a finite strictly positive real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _validate_finite_scalar(value: Any, *, name: str) -> float:
    """Validate a finite real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_nonnegative_scalar(value: Any, *, name: str) -> float:
    """Validate a finite nonnegative real scalar."""
    result = _validate_finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _validate_node_count(n_nodes: Any) -> int:
    """Validate a positive integer node count."""
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(
        n_nodes, Integral
    ):
        raise TypeError("n_nodes must be an integer.")
    result = int(n_nodes)
    if result < 1:
        raise ValueError("n_nodes must be at least 1.")
    return result


def _prepare_vector(
    values: Optional[Any],
    *,
    n_nodes: int,
    name: str,
    default: np.ndarray,
) -> np.ndarray:
    """Validate one input/readout vector or use a supplied default."""
    if values is None:
        return default.copy()

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != n_nodes:
        raise ValueError(
            f"{name} must be a one-dimensional vector of length {n_nodes}."
        )
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"Complex-valued {name} is not supported.")

    try:
        vector = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return vector


def _prepare_input_output_vectors(
    n_nodes: int,
    B: Optional[Any],
    C: Optional[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return input and readout vectors, using normalized uniform defaults."""
    uniform = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)
    input_vector = _prepare_vector(
        B,
        n_nodes=n_nodes,
        name="B",
        default=uniform,
    )
    readout_vector = _prepare_vector(
        C,
        n_nodes=n_nodes,
        name="C",
        default=uniform,
    )
    return input_vector, readout_vector


def _identity(n_nodes: int, *, sparse: bool) -> Any:
    """Construct a dense or sparse identity matrix."""
    if sparse:
        from scipy import sparse as scipy_sparse

        return scipy_sparse.eye(n_nodes, format="csr", dtype=np.float64)
    return np.eye(n_nodes, dtype=np.float64)


def _solve_linear_system(matrix: Any, rhs: np.ndarray) -> np.ndarray:
    """Solve a finite dense or sparse linear system with clear errors."""
    if is_sparse_matrix(matrix):
        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution = spsolve(matrix, rhs)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise np.linalg.LinAlgError(
                "The requested linear system is singular or could not be "
                "solved."
            ) from exc
    else:
        try:
            solution = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "The requested linear system is singular or could not be "
                "solved."
            ) from exc

    result = np.asarray(solution, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise np.linalg.LinAlgError(
            "The linear-system solution contains non-finite values."
        )
    return result


def _handle_undefined_cutoff(
    *,
    zero_overlap: UndefinedCutoffPolicy,
    reason: str,
) -> tuple[float, float, bool, bool]:
    """Apply the requested policy for an undefined cutoff."""
    if zero_overlap == "raise":
        raise ValueError(reason)
    return float("nan"), float("nan"), False, False


def network_cutoff_time(
    W: Any,
    tau_node: float,
    *,
    asymptotic_order: float = 1.0,
    coupling: float = 1.0,
    B: Optional[Any] = None,
    C: Optional[Any] = None,
    gain_tolerance: float = 1e-12,
    zero_overlap: UndefinedCutoffPolicy = "raise",
) -> NetworkCutoffResult:
    """Calculate the paper-style cutoff time from the full matrix resolvent.

    This implements the paper's zero-frequency/high-frequency asymptote
    definition.  It assumes

    ``h(0) = tau_node**asymptotic_order``

    and ``h(s) ~ 1/s**asymptotic_order`` at high frequency.  Only the zero-
    frequency value is needed, so this function does not determine whether an
    otherwise unspecified node filter is dynamically stable.  For the fully
    specified first-order exponential model, use
    :func:`exponential_network_timescales`.

    Parameters
    ----------
    W:
        Square recurrent adjacency/weight matrix.
    tau_node:
        Cutoff time of an isolated node.
    asymptotic_order:
        Positive high-frequency roll-off order ``g``.
    coupling:
        Scalar multiplying ``W``.
    B, C:
        Optional input and readout vectors.  Both default to
        ``ones(N)/sqrt(N)``.
    gain_tolerance:
        Threshold below which the high-frequency or DC gain is treated as
        zero.
    zero_overlap:
        ``"raise"`` rejects an undefined normalization; ``"nan"`` returns
        NaN diagnostics instead.
    """
    tau_node = _validate_positive_scalar(tau_node, name="tau_node")
    g = _validate_positive_scalar(
        asymptotic_order,
        name="asymptotic_order",
    )
    coupling = _validate_finite_scalar(coupling, name="coupling")
    gain_tolerance = _validate_nonnegative_scalar(
        gain_tolerance,
        name="gain_tolerance",
    )
    if zero_overlap not in ("nan", "raise"):
        raise ValueError("zero_overlap must be either 'nan' or 'raise'.")

    matrix, n_nodes = prepare_adjacency(W)
    input_vector, readout_vector = _prepare_input_output_vectors(
        n_nodes,
        B,
        C,
    )

    h_zero = tau_node**g
    sparse = is_sparse_matrix(matrix)
    resolvent = _identity(n_nodes, sparse=sparse) - (
        h_zero * coupling * matrix
    )
    propagated = _solve_linear_system(resolvent, input_vector)

    dc_gain = float(h_zero * np.dot(readout_vector, propagated))
    instantaneous_gain = float(np.dot(readout_vector, input_vector))

    if abs(instantaneous_gain) <= gain_tolerance:
        cutoff_time, normalized, defined, positive = _handle_undefined_cutoff(
            zero_overlap=zero_overlap,
            reason=(
                "C.T @ B is zero or too small. The leading high-frequency "
                "term vanishes, so the paper's cutoff normalization is not "
                "defined."
            ),
        )
    elif abs(dc_gain) <= gain_tolerance:
        cutoff_time, normalized, defined, positive = _handle_undefined_cutoff(
            zero_overlap=zero_overlap,
            reason=(
                "The DC gain is zero or too small. A cutoff based on the "
                "low-frequency baseline is not defined."
            ),
        )
    else:
        normalized = dc_gain / instantaneous_gain
        cutoff_time = abs(normalized) ** (1.0 / g)
        defined = True
        positive = normalized > 0.0

    return {
        "cutoff_time": float(cutoff_time),
        "cutoff_defined": bool(defined),
        "dc_gain": dc_gain,
        "instantaneous_gain": instantaneous_gain,
        "normalized_dc_gain": float(normalized),
        "positive_normalized_dc_gain": bool(positive),
        "asymptotic_order": g,
    }


def paper_cutoff_time_constant(
    W: Any,
    tau_node: float,
    *,
    g: float = 1.0,
    coupling: float = 1.0,
    B: Optional[Any] = None,
    C: Optional[Any] = None,
    gain_tolerance: float = 1e-12,
) -> float:
    """Return only the full-matrix paper-style cutoff time.

    This is a convenience wrapper around :func:`network_cutoff_time` using
    ``asymptotic_order=g`` and strict handling of an undefined normalization.
    """
    return network_cutoff_time(
        W,
        tau_node,
        asymptotic_order=g,
        coupling=coupling,
        B=B,
        C=C,
        gain_tolerance=gain_tolerance,
        zero_overlap="raise",
    )["cutoff_time"]


def motif_cutoff_times_from_cumulants(
    chain_cumulants: Any,
    *,
    n_nodes: int,
    tau_node: float,
    g: float = 1.0,
    coupling: float = 1.0,
    invalid_denominator: InvalidDenominatorPolicy = "magnitude",
) -> MotifTimescaleResult:
    """Calculate successive cutoff estimates from precomputed cumulants.

    The order-``n`` feedback contribution is

    ``N**n * kappa_n * coupling**n * tau_node**(n*g)``.
    """
    cumulants = validate_real_vector(
        chain_cumulants,
        name="chain_cumulants",
    )
    n_nodes = _validate_node_count(n_nodes)
    tau_node = _validate_positive_scalar(tau_node, name="tau_node")
    g = _validate_positive_scalar(g, name="g")
    coupling = _validate_finite_scalar(coupling, name="coupling")
    if invalid_denominator not in ("magnitude", "nan", "raise"):
        raise ValueError(
            "invalid_denominator must be 'magnitude', 'nan', or 'raise'."
        )

    max_order = int(cumulants.size)
    orders = np.arange(1, max_order + 1, dtype=int)
    dimensionless_scale = n_nodes * coupling * tau_node**g

    with np.errstate(over="ignore", invalid="ignore"):
        feedback_terms = cumulants * np.power(
            dimensionless_scale,
            orders,
        )
        cumulative_feedback = np.cumsum(feedback_terms)
        denominator = 1.0 - cumulative_feedback

    paper_valid = np.isfinite(denominator) & (denominator > 0.0)
    if invalid_denominator == "raise" and not np.all(paper_valid):
        first_bad = int(np.flatnonzero(~paper_valid)[0]) + 1
        raise ValueError(
            "The motif denominator is nonpositive or non-finite at order "
            f"{first_bad}; the paper's positive-gain time-constant formula "
            "is not valid there."
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        time_constants = tau_node / np.abs(denominator) ** (1.0 / g)
    if invalid_denominator == "nan":
        time_constants = np.where(paper_valid, time_constants, np.nan)

    # Several aliases are intentionally returned.  They make the result easy
    # to read while preserving the API used by earlier versions of the package.
    return {
        "order": orders,
        "chain_cumulants": cumulants.copy(),
        "feedback_terms": feedback_terms,
        "contributions": feedback_terms,
        "cumulative_feedback": cumulative_feedback,
        "denominator": denominator,
        "denominators": denominator,
        "time_constants": time_constants,
        "cutoff_times": time_constants,
        "paper_valid": paper_valid,
        "valid": paper_valid,
    }


def motif_cutoff_times_by_order(
    W: Any,
    max_order: int,
    tau_node: float,
    *,
    g: float = 1.0,
    coupling: float = 1.0,
    method: CumulantMethod = "projector",
    invalid_denominator: InvalidDenominatorPolicy = "magnitude",
    include_exact: bool = True,
) -> MotifTimescaleResult:
    """Calculate the cutoff-time approximation after every motif order.

    This function uses the paper's default uniform input and readout.  When
    ``include_exact=True``, it also evaluates the full matrix resolvent and
    reports each truncation's relative error.
    """
    max_order = validate_max_order(max_order)
    tau_node = _validate_positive_scalar(tau_node, name="tau_node")
    g = _validate_positive_scalar(g, name="g")
    coupling = _validate_finite_scalar(coupling, name="coupling")
    if method not in ("projector", "moments"):
        raise ValueError("method must be either 'projector' or 'moments'.")
    if not isinstance(include_exact, (bool, np.bool_)):
        raise TypeError("include_exact must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    cumulants = chain_motif_cumulants(
        matrix,
        max_order=max_order,
        method=method,
        return_moments=False,
    )["cumulants"]

    result = motif_cutoff_times_from_cumulants(
        cumulants,
        n_nodes=n_nodes,
        tau_node=tau_node,
        g=g,
        coupling=coupling,
        invalid_denominator=invalid_denominator,
    )

    if include_exact:
        exact = network_cutoff_time(
            matrix,
            tau_node,
            asymptotic_order=g,
            coupling=coupling,
            zero_overlap="raise",
        )["cutoff_time"]
        times = result["time_constants"]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = np.abs(times - exact) / abs(exact)
        result["exact_time_constant"] = exact
        result["full_time_constant"] = exact
        result["exact_cutoff_time"] = exact
        result["relative_error"] = relative_error

    return result


def _system_matrix(
    matrix: Any,
    n_nodes: int,
    *,
    tau_node: float,
    coupling: float,
) -> Any:
    """Construct ``coupling * W - I/tau_node``."""
    return coupling * matrix - _identity(
        n_nodes,
        sparse=is_sparse_matrix(matrix),
    ) / tau_node


def _spectral_abscissa_and_optional_poles(
    system_matrix: Any,
    n_nodes: int,
    *,
    return_poles: bool,
) -> tuple[float, Optional[np.ndarray]]:
    """Calculate the rightmost pole, retaining sparse operations when possible."""
    if not is_sparse_matrix(system_matrix):
        poles = np.linalg.eigvals(np.asarray(system_matrix, dtype=np.float64))
        return float(np.max(poles.real)), poles if return_poles else None

    if return_poles:
        warnings.warn(
            "return_poles=True converts a sparse system matrix to dense. "
            "For large networks, leave return_poles=False.",
            RuntimeWarning,
            stacklevel=3,
        )
        poles = np.linalg.eigvals(system_matrix.toarray())
        return float(np.max(poles.real)), poles

    if n_nodes <= 3:
        poles = np.linalg.eigvals(system_matrix.toarray())
        return float(np.max(poles.real)), None

    from scipy.sparse.linalg import ArpackNoConvergence, eigs

    try:
        rightmost = eigs(
            system_matrix,
            k=1,
            which="LR",
            return_eigenvectors=False,
        )
    except ArpackNoConvergence as exc:  # pragma: no cover - ARPACK dependent.
        if exc.eigenvalues is not None and exc.eigenvalues.size:
            rightmost = exc.eigenvalues
        elif n_nodes <= 512:
            poles = np.linalg.eigvals(system_matrix.toarray())
            return float(np.max(poles.real)), None
        else:
            raise RuntimeError(
                "ARPACK did not converge while estimating the rightmost pole."
            ) from exc

    return float(np.max(np.asarray(rightmost).real)), None


def exponential_network_timescales(
    W: Any,
    tau_node: float,
    *,
    coupling: float = 1.0,
    B: Optional[Any] = None,
    C: Optional[Any] = None,
    stability_tolerance: float = 1e-12,
    gain_tolerance: float = 1e-12,
    zero_overlap: UndefinedCutoffPolicy = "nan",
    raise_on_unstable: bool = True,
    require_stable: Optional[bool] = None,
    return_poles: bool = False,
) -> ExponentialTimescaleResult:
    """Calculate exact cutoff and dominant-pole times for exponential nodes.

    The generator is ``M = coupling*W - I/tau_node``.  A stable network has
    ``max(real(eig(M))) < 0``.

    ``paper_cutoff_time``
        Magnitude of the integrated scalar impulse response divided by its
        instantaneous gain ``C.T @ B``.

    ``dominant_pole_time``
        ``-1/max(real(eig(M)))`` for a stable system.  This is system-wide and
        may correspond to a mode not visible for the selected ``B`` and ``C``.

    ``require_stable`` is retained as a backward-compatible alias for
    ``raise_on_unstable``.
    """
    tau_node = _validate_positive_scalar(tau_node, name="tau_node")
    coupling = _validate_finite_scalar(coupling, name="coupling")
    stability_tolerance = _validate_nonnegative_scalar(
        stability_tolerance,
        name="stability_tolerance",
    )
    gain_tolerance = _validate_nonnegative_scalar(
        gain_tolerance,
        name="gain_tolerance",
    )
    if zero_overlap not in ("nan", "raise"):
        raise ValueError("zero_overlap must be either 'nan' or 'raise'.")
    if not isinstance(raise_on_unstable, (bool, np.bool_)):
        raise TypeError("raise_on_unstable must be a Boolean value.")
    if require_stable is not None:
        if not isinstance(require_stable, (bool, np.bool_)):
            raise TypeError("require_stable must be a Boolean value or None.")
        raise_on_unstable = bool(require_stable)
    if not isinstance(return_poles, (bool, np.bool_)):
        raise TypeError("return_poles must be a Boolean value.")

    matrix, n_nodes = prepare_adjacency(W)
    input_vector, readout_vector = _prepare_input_output_vectors(
        n_nodes,
        B,
        C,
    )
    generator = _system_matrix(
        matrix,
        n_nodes,
        tau_node=tau_node,
        coupling=coupling,
    )

    spectral_abscissa, poles = _spectral_abscissa_and_optional_poles(
        generator,
        n_nodes,
        return_poles=bool(return_poles),
    )
    stable = spectral_abscissa < -stability_tolerance

    if raise_on_unstable and not stable:
        raise ValueError(
            "The exponential-node network is not asymptotically stable: "
            f"maximum real pole = {spectral_abscissa:.12g}."
        )

    if not stable:
        cutoff_time = float("nan")
        cutoff_defined = False
        dc_gain = float("nan")
        instantaneous_gain = float(np.dot(readout_vector, input_vector))
        normalized_dc_gain = float("nan")
        positive_normalized_dc_gain = False
        if abs(spectral_abscissa) <= stability_tolerance:
            dominant_pole_time = float("inf")
        else:
            dominant_pole_time = float("nan")
    else:
        integrated_state = _solve_linear_system(-generator, input_vector)
        dc_gain = float(np.dot(readout_vector, integrated_state))
        instantaneous_gain = float(np.dot(readout_vector, input_vector))

        if abs(instantaneous_gain) <= gain_tolerance:
            (
                cutoff_time,
                normalized_dc_gain,
                cutoff_defined,
                positive_normalized_dc_gain,
            ) = _handle_undefined_cutoff(
                zero_overlap=zero_overlap,
                reason=(
                    "C.T @ B is zero or too small, so the normalized "
                    "impulse-response area is not defined."
                ),
            )
        elif abs(dc_gain) <= gain_tolerance:
            (
                cutoff_time,
                normalized_dc_gain,
                cutoff_defined,
                positive_normalized_dc_gain,
            ) = _handle_undefined_cutoff(
                zero_overlap=zero_overlap,
                reason=(
                    "The integrated impulse response is zero or too small, "
                    "so the cutoff time is not defined."
                ),
            )
        else:
            normalized_dc_gain = dc_gain / instantaneous_gain
            cutoff_time = abs(normalized_dc_gain)
            cutoff_defined = True
            positive_normalized_dc_gain = normalized_dc_gain > 0.0

        dominant_pole_time = -1.0 / spectral_abscissa

    result: ExponentialTimescaleResult = {
        "cutoff_time": float(cutoff_time),
        "paper_cutoff_time": float(cutoff_time),
        "cutoff_defined": bool(cutoff_defined),
        "dominant_pole_time": float(dominant_pole_time),
        "dc_gain": float(dc_gain),
        "dc_response": float(dc_gain),
        "instantaneous_gain": float(instantaneous_gain),
        "instantaneous_response": float(instantaneous_gain),
        "normalized_dc_gain": float(normalized_dc_gain),
        "signed_normalized_impulse_area": float(normalized_dc_gain),
        "positive_normalized_dc_gain": bool(positive_normalized_dc_gain),
        "spectral_abscissa": float(spectral_abscissa),
        "stability_margin": float(-spectral_abscissa),
        "stable": bool(stable),
    }
    if poles is not None:
        result["poles"] = poles
    return result


def exponential_impulse_response(
    W: Any,
    times: Any,
    tau_node: float,
    *,
    coupling: float = 1.0,
    B: Optional[Any] = None,
    C: Optional[Any] = None,
) -> np.ndarray:
    """Evaluate ``C.T exp((coupling*W - I/tau_node)t) B``.

    SciPy is required.  The implementation uses ``expm_multiply`` rather than
    explicitly forming a matrix exponential and works with dense or sparse
    matrices.
    """
    tau_node = _validate_positive_scalar(tau_node, name="tau_node")
    coupling = _validate_finite_scalar(coupling, name="coupling")

    raw_times = np.asarray(times)
    if raw_times.ndim == 0:
        raw_times = raw_times.reshape(1)
    if raw_times.ndim != 1 or raw_times.size == 0:
        raise ValueError(
            "times must be a scalar or a nonempty one-dimensional sequence."
        )
    if np.issubdtype(raw_times.dtype, np.complexfloating):
        raise TypeError("Complex-valued times are not supported.")
    try:
        time_array = raw_times.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("times must contain numeric values.") from exc
    if not np.all(np.isfinite(time_array)):
        raise ValueError("times contains NaN or infinite values.")
    if np.any(time_array < 0.0):
        raise ValueError("times must be nonnegative.")

    try:
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "exponential_impulse_response requires SciPy>=1.9. Install "
            "the package with the 'dynamics' optional extra."
        ) from exc

    matrix, n_nodes = prepare_adjacency(W)
    input_vector, readout_vector = _prepare_input_output_vectors(
        n_nodes,
        B,
        C,
    )
    generator = _system_matrix(
        matrix,
        n_nodes,
        tau_node=tau_node,
        coupling=coupling,
    )

    if is_sparse_matrix(generator):
        trace_generator = float(np.asarray(generator.diagonal()).sum())
    else:
        trace_generator = float(np.trace(generator))

    evenly_spaced = False
    if time_array.size >= 2 and np.all(np.diff(time_array) >= 0.0):
        differences = np.diff(time_array)
        evenly_spaced = differences[0] > 0.0 and np.allclose(
            differences,
            differences[0],
            rtol=1e-12,
            atol=1e-15,
        )

    if evenly_spaced:
        states = expm_multiply(
            generator,
            input_vector,
            start=float(time_array[0]),
            stop=float(time_array[-1]),
            num=int(time_array.size),
            endpoint=True,
            traceA=trace_generator,
        )
        response = np.asarray(states) @ readout_vector
    else:
        response = np.empty(time_array.size, dtype=np.float64)
        for index, time in enumerate(time_array):
            state = expm_multiply(
                generator * float(time),
                input_vector,
                traceA=trace_generator * float(time),
            )
            response[index] = float(np.dot(readout_vector, state))

    return np.asarray(np.real_if_close(response), dtype=np.float64)


def _spectral_radius(matrix: Any, n_nodes: int) -> float:
    """Calculate the spectral radius with sparse support."""
    if not is_sparse_matrix(matrix):
        eigenvalues = np.linalg.eigvals(np.asarray(matrix, dtype=np.float64))
        return float(np.max(np.abs(eigenvalues)))

    if n_nodes <= 3:
        eigenvalues = np.linalg.eigvals(matrix.toarray())
        return float(np.max(np.abs(eigenvalues)))

    from scipy.sparse.linalg import ArpackNoConvergence, eigs

    try:
        leading = eigs(
            matrix,
            k=1,
            which="LM",
            return_eigenvectors=False,
        )
    except ArpackNoConvergence as exc:  # pragma: no cover - ARPACK dependent.
        if exc.eigenvalues is not None and exc.eigenvalues.size:
            leading = exc.eigenvalues
        elif n_nodes <= 512:
            eigenvalues = np.linalg.eigvals(matrix.toarray())
            return float(np.max(np.abs(eigenvalues)))
        else:
            raise RuntimeError(
                "ARPACK did not converge while estimating the spectral radius."
            ) from exc

    return float(np.max(np.abs(np.asarray(leading))))


def structural_timescale_curve(
    W: Any,
    eta: Any,
    *,
    B: Optional[Any] = None,
    C: Optional[Any] = None,
    gain_tolerance: float = 1e-12,
) -> StructuralTimescaleResult:
    """Calculate a scale-normalized, W-only comparative timescale curve.

    For a nonnegative adjacency matrix, this sets

    ``coupling * tau_node = eta / spectral_radius(W)``

    with ``tau_node=1`` and returns

    ``R_eta = |C.T (I - eta W/rho(W))^-1 B / (C.T B)|``.

    ``R_eta`` is dimensionless.  It compares network structures at the same
    fraction ``eta`` of the spectral instability threshold; it is not a
    calibrated physical time constant.
    """
    matrix, n_nodes = prepare_adjacency(W)
    if is_sparse_matrix(matrix):
        if np.any(matrix.data < 0.0):
            raise ValueError(
                "structural_timescale_curve requires a nonnegative matrix."
            )
    elif np.any(matrix < 0.0):
        raise ValueError(
            "structural_timescale_curve requires a nonnegative matrix."
        )

    raw_eta = np.asarray(eta)
    if raw_eta.ndim == 0:
        raw_eta = raw_eta.reshape(1)
    if raw_eta.ndim != 1 or raw_eta.size == 0:
        raise ValueError("eta must be a scalar or nonempty 1D sequence.")
    if np.issubdtype(raw_eta.dtype, np.complexfloating):
        raise TypeError("Complex-valued eta is not supported.")
    try:
        eta_array = raw_eta.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("eta must contain numeric values.") from exc
    if not np.all(np.isfinite(eta_array)):
        raise ValueError("eta contains NaN or infinite values.")
    if np.any(eta_array < 0.0) or np.any(eta_array >= 1.0):
        raise ValueError("Every eta value must satisfy 0 <= eta < 1.")

    gain_tolerance = _validate_nonnegative_scalar(
        gain_tolerance,
        name="gain_tolerance",
    )
    input_vector, readout_vector = _prepare_input_output_vectors(
        n_nodes,
        B,
        C,
    )
    overlap = float(np.dot(readout_vector, input_vector))
    if abs(overlap) <= gain_tolerance:
        raise ValueError(
            "C.T @ B is zero or too small, so the structural cutoff ratio "
            "is not defined."
        )

    radius = _spectral_radius(matrix, n_nodes)
    if not np.isfinite(radius) or radius <= gain_tolerance:
        raise ValueError(
            "W must have a positive spectral radius for eta normalization."
        )

    sparse = is_sparse_matrix(matrix)
    identity = _identity(n_nodes, sparse=sparse)
    cutoff_ratio = np.empty(eta_array.size, dtype=np.float64)
    effective_coupling = eta_array / radius

    for index, beta in enumerate(effective_coupling):
        propagated = _solve_linear_system(
            identity - float(beta) * matrix,
            input_vector,
        )
        normalized = float(np.dot(readout_vector, propagated)) / overlap
        cutoff_ratio[index] = abs(normalized)

    return {
        "eta": eta_array.copy(),
        "spectral_radius": radius,
        "effective_coupling": effective_coupling,
        "cutoff_ratio": cutoff_ratio,
    }
