# Motif cumulants, directed motifs, and network timescales

A Python package for analyzing a directed adjacency matrix at three distinct
levels:

1. **Analytic weighted-walk cumulants**: chain, PRE closed-walk cycle,
   divergent, convergent, and PLOS mixed-trace families.
2. **Classical finite-node motifs**: exact SONET second-order counts and the
   16-class induced directed-triad census.
3. **Functional and comparative analyses**: null-model enrichment,
   input/readout-weighted and population-resolved cumulants, and motif-based
   network response times.

The implementation keeps these definitions separate because a weighted walk,
an induced subgraph, and a motif enrichment z-score are not interchangeable.

Current package version: **0.6.0**.

Primary references:

> Y. Hu, S. L. Brunton, N. Cain, S. Mihalas, J. N. Kutz, and
> E. Shea-Brown, *Feedback through graph motifs relates structure and function
> in complex networks*, Physical Review E 98, 062312 (2018).
> DOI: 10.1103/PhysRevE.98.062312

> S. Recanatesi, G. K. Ocker, M. A. Buice, and E. Shea-Brown,
> *Dimensionality in recurrent spiking networks: Global trends in activity and
> local origins in connectivity*, PLOS Computational Biology 15(7), e1006446
> (2019). DOI: 10.1371/journal.pcbi.1006446

> L. Zhao, B. Beverlin II, T. Netoff, and D. Q. Nykamp,
> *Synchronization from Second Order Network Connectivity Statistics*,
> Frontiers in Computational Neuroscience 5:28 (2011).
> DOI: 10.3389/fncom.2011.00028

> Y. Hu, J. Trousdale, K. Josić, and E. Shea-Brown,
> *Local paths to global coherence: Cutting networks down to size*,
> Physical Review E 89, 032802 (2014).
> DOI: 10.1103/PhysRevE.89.032802

> R. Milo et al., *Network motifs: Simple building blocks of complex
> networks*, Science 298, 824-827 (2002).
> DOI: 10.1126/science.298.5594.824

The adjacency convention is

```text
W[i, j] = weight of the directed edge j -> i
```

Dense NumPy-compatible matrices are supported throughout. SciPy sparse
matrices are also supported when SciPy is installed.

## Installation

From this directory:

```bash
python -m pip install .
```

For sparse matrices or direct impulse-response evaluation:

```bash
python -m pip install ".[sparse]"
```

The equivalent dynamics extra is also available:

```bash
python -m pip install ".[dynamics]"
```

To install the optional validation dependencies used by the complete test
suite:

```bash
python -m pip install ".[test]"
```

For development and validation dependencies:

```bash
python -m pip install ".[test]"
```

## Motif definitions

Let `N` be the number of nodes,

```text
e = ones(N) / sqrt(N)
Theta = I - e e^T.
```

### Chain motifs

The order-`n` chain moment and cumulant are

```text
mu_chain_n    = sum(W^n) / N^(n+1)
kappa_chain_n = e^T W (Theta W)^(n-1) e / N^n.
```

The equivalent ordered-composition recurrence is

```text
kappa_chain_n = mu_chain_n
                - sum_{j=1}^{n-1} kappa_chain_j * mu_chain_(n-j).
```

### Cycle motifs

The order-`n` cycle moment and cumulant are

```text
mu_cycle_n    = trace(W^n) / N^n
kappa_cycle_n = trace((Theta W)^n) / N^n.
```

The paper also relates a cycle moment to the chain cumulants:

```text
mu_cycle_n = kappa_cycle_n
             + sum_{(n1,...,nt) in C(n)}
                   (n/t) * product_i kappa_chain_ni,
```

where `C(n)` is the set of ordered compositions of `n`. The package evaluates
this relation using an `O(K^2)` recurrence instead of enumerating all
`2^(n-1)` compositions.

These formulas count weighted directed walks. For example, `trace(W^n)` counts
closed walks and does not require all nodes or edges in a walk to be distinct.

### Divergent motifs

A divergent `(n, m)` motif contains two paths of lengths `n` and `m` leaving a
common source. With `A = W/N`, its moment is

```text
mu_div[n,m] = e^T A^n (A^T)^m e.
```

Let

```text
B_n = (A Theta)^(n-1) A.
```

The irreducible divergent cumulant is

```text
kappa_div[n,m] = e^T B_n Theta B_m^T e.
```

Use:

```python
from motif_cumulants import divergent_motif_cumulants

divergent = divergent_motif_cumulants(W, max_order=4)
print(divergent["cumulants"][1, 0])  # kappa_div_(2,1)
```

Rows and columns correspond to branch lengths `1, ..., max_order`. The total
number of edges is available as `divergent["total_order"]`.

### Convergent motifs

A convergent `(n, m)` motif contains two paths arriving at a common target:

```text
mu_conv[n,m]    = e^T (A^T)^n A^m e
kappa_conv[n,m] = e^T B_n^T Theta B_m e.
```

Use:

```python
from motif_cumulants import convergent_motif_cumulants

convergent = convergent_motif_cumulants(W, max_order=4)
print(convergent["cumulants"][0, 0])  # kappa_conv_(1,1)
```

At `(1, 1)`, divergent motifs quantify excess shared output from a source,
while convergent motifs quantify excess shared input to a target.

### Mixed trace motifs

A mixed trace `(n, m)` motif consists of two paths with the same ordered
starting and ending nodes. The paper-normalized moment is

```text
mu_trace[n,m] = trace(W^n (W^T)^m) / N^(n+m+1).
```

For example, `(2, 1)` detects a two-step path plus a direct edge connecting the
same source and target, which is a weighted feed-forward-loop statistic. Its
cumulant is

```text
kappa_trace[n,m]
    = trace(B_n Theta B_m^T Theta) / N,
```

where `B_n` above already contains the factors `W/N`.

Use:

```python
from motif_cumulants import mixed_trace_motif_cumulants

trace = mixed_trace_motif_cumulants(W, max_order=4)
print(trace["moments"][1, 0])     # mu_trace_(2,1)
print(trace["cumulants"][1, 0])  # kappa_trace_(2,1)
```

The default `normalization="recanatesi"` includes the extra `1/N` used in the
PLOS paper. To compare magnitudes with this package's cycle statistics, use
`normalization="cycle_compatible"`, which multiplies the mixed trace values by
`N`.

Mixed trace cumulants require the dense projector `Theta`; sparse inputs are
therefore converted to dense with a warning. Mixed trace *moments* retain
sparse matrix operations.

### Weighted walk-based second-order profile

The weighted walk-based second-order profile consists of chain, divergent,
convergent, and reciprocal matrix statistics. The function below returns their
raw moments and independent-edge excesses relative to `p^2`. Repeated node
indices are allowed here; use `sonet_motif_statistics` in the next section for
exact distinct-node SONET counts:

```python
from motif_cumulants import second_order_motif_statistics

profile = second_order_motif_statistics(W, remove_self_loops=True)
for name, moment, cumulant in zip(
    profile["motif"],
    profile["moments"],
    profile["cumulants"],
):
    print(name, moment, cumulant)
```

For the reciprocal row,

```text
mu_reciprocal = trace(W^2) / N^2
kappa_reciprocal_SONET = mu_reciprocal - p^2.
```

This SONET reciprocal excess is **not generally identical** to the order-2
cycle cumulant from Hu et al. The result therefore also reports
`cycle_order_2_cumulant` explicitly.

### Exact finite-size SONET counts

`second_order_motif_statistics` uses the package's weighted walk
normalization and permits repeated indices. For an exact loop-free binary
network calculation on distinct node positions, use:

```python
from motif_cumulants import sonet_motif_statistics

sonet = sonet_motif_statistics(
    W,
    threshold=0.0,
    edge_presence="nonzero",  # abs(W) > threshold
)

for name, count, frequency, alpha in zip(
    sonet["motif"],
    sonet["counts"],
    sonet["frequencies"],
    sonet["alpha"],
):
    print(name, count, frequency, alpha)
```

The exact counts are:

```text
chain      = sum_i in_degree[i] * out_degree[i]
             - 2 * reciprocal_count
divergent  = sum_i choose(out_degree[i], 2)
convergent = sum_i choose(in_degree[i], 2)
reciprocal = number of unordered mutual dyads.
```

The subtraction in the chain count removes two-node backtracking paths. Each
frequency is divided by its exact number of distinct-node placements. With
edge density `p`, `alpha = frequency / p^2 - 1`. A negative signed weight is
retained by `edge_presence="nonzero"`; use `"positive"` to ignore it.

### Unified covariance-motif interface

The PLOS covariance expansion uses chain, divergent, convergent, and mixed
trace families together. They can be calculated in one call:

```python
from motif_cumulants import covariance_motif_cumulants

covariance_motifs = covariance_motif_cumulants(
    W,
    max_order=4,
    include_trace=True,
    include_second_order=True,
)

print(covariance_motifs["chain"]["cumulants"])
print(covariance_motifs["divergent"]["cumulants"])
print(covariance_motifs["convergent"]["cumulants"])
print(covariance_motifs["trace"]["cumulants"])
```

`include_trace=False` avoids the dense projector required by mixed trace
cumulants. `include_second_order=False` omits the compact convenience profile.
The structural values are computable from `W`; predicting an actual covariance
matrix or activity dimension additionally requires a response model and input
or noise covariance.

### Input/readout-specific chain cumulants

Uniform averaging can hide which paths are relevant to an experiment. The
PRE supplementary construction for arbitrary input `B` and readout `C` is
available through:

```python
from motif_cumulants import generalized_chain_motif_cumulants

specific = generalized_chain_motif_cumulants(
    W,
    max_order=6,
    B=B,
    C=C,
)

print(specific["overlap"])      # C.T @ B
print(specific["cumulants"])
```

The aliases `input_output_chain_motif_cumulants` and
`weighted_chain_motif_cumulants` call the same implementation. The normalized
formula requires `C.T @ B != 0`. Divergent and convergent calculations also
accept a nonuniform `weights=` vector and normalize it internally.

### Population-resolved cumulants

Known node classes can be retained rather than collapsed into one scalar:

```python
from motif_cumulants import population_motif_cumulants

groups = np.array(["E", "E", "I", "I"])
population = population_motif_cumulants(W, groups, max_order=3)

print(population["group"])
print(population["cumulants"]["chain"])
print(population["cumulants"]["divergent"])
print(population["cumulants"]["convergent"])
```

For chain motifs, `cumulants[n-1, target_group, source_group]` records the
order-`n` contribution from the source population to the target population.
The implementation uses the block projector described by Hu et al. (2014),
DOI `10.1103/PhysRevE.89.032802`.

### Motif cumulants versus induced-subgraph motifs

The calculations above are matrix/walk statistics designed to enter analytic
network-dynamics expansions. They permit repeated node indices and naturally
support weighted matrices. They are different from an induced triad census,
where each set of three distinct nodes is assigned to one of the directed
triad isomorphism classes and compared with a null-network ensemble. A triad
census and motif z-scores are useful complementary analyses, but they should
not be substituted directly into the transfer-function or covariance
resummation formulas.

### Exact induced directed triads

Each unordered set of three distinct nodes can be classified into one of the
16 standard directed triads:

```python
from motif_cumulants import directed_triad_census

triads = directed_triad_census(
    W,
    threshold=0.0,
    edge_presence="nonzero",
)

print(triads["triad"])
print(triads["counts"])
print(triads["proportions"])
print(triads["count_by_name"]["030T"])  # transitive/feed-forward triad
print(triads["count_by_name"]["030C"])  # directed three-cycle
```

The census is induced: absent edges inside the selected three-node set matter,
and all 16 counts sum to `choose(N, 3)`. Self-loops are ignored. The exact
implementation is `O(N^3)`.

### Null-model enrichment

A classical network motif is an excess or deficit relative to a specified null
ensemble. The package provides:

- `density_matched_null`: exact total directed edge count;
- `directed_degree_preserving_null`: exact binary in- and out-degree sequences;
- `block_density_matched_null`: exact edge counts in every population block;
- `shuffle_edge_weights`: fixed topology with shuffled nonzero weights.

```python
from motif_cumulants import triad_enrichment

enrichment = triad_enrichment(
    W,
    n_random=500,
    null_model="degree",
    random_state=0,
)

print(enrichment["z_score"])
print(enrichment["empirical_p_two_sided"])
```

The convenience interface below uses longer null-model names and additionally
returns a normalized triad significance profile:

```python
from motif_cumulants import triad_motif_enrichment

profile = triad_motif_enrichment(
    W,
    n_random=500,
    null_model="degree_preserving",
    random_state=0,
)
print(profile["z_scores"])
print(profile["significance_profile"])
```

The degree-preserving null uses directed double-edge swaps. A warning indicates
that a constrained graph admitted fewer valid swaps than requested; increase
`max_tries`, lower `n_swaps`, or inspect whether the degree sequence is nearly
rigid. Null-model choice is part of the scientific hypothesis, not merely a
computational setting.

## Basic motif use

```python
import numpy as np
from motif_cumulants import (
    chain_motif_cumulants,
    covariance_motif_cumulants,
    cycle_motif_cumulants,
    directed_triad_census,
    sonet_motif_statistics,
)

W = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

chain = chain_motif_cumulants(W, max_order=6)
cycle = cycle_motif_cumulants(W, max_order=6)
covariance = covariance_motif_cumulants(W, max_order=3)
sonet = sonet_motif_statistics(W)
triads = directed_triad_census(W)

print(chain["cumulants"])
print(cycle["cumulants"])
print(covariance["divergent"]["cumulants"])
print(covariance["trace"]["cumulants"])
print(sonet["motif"], sonet["alpha"])
print(triads["count_by_name"])
```

## What is required to calculate a time constant?

A bare binary or arbitrarily normalized adjacency matrix does **not** determine
an absolute time in seconds. The paper's network transfer function is

```text
G(s) = C^T (I - h(s) W)^(-1) B h(s),
```

so a physical response time also requires:

- the single-node temporal filter `h(s)`, or at least its node time constant;
- the physical scale of recurrent connectivity;
- input and readout vectors `B` and `C` when they are not uniform.

The package represents a separate global recurrent scale with `coupling`. If
`W` already contains the physical connection strengths, leave `coupling=1`.

## Full-matrix frequency-cutoff time

The paper defines a time constant from the intersection of the low-frequency
Bode magnitude baseline and the high-frequency asymptote. Suppose

```text
h(s) ~ 1/s^g
```

at high frequency and an isolated node has cutoff time `tau_node`. Then

```text
h(0) = tau_node^g.
```

For the paper's default uniform input and readout,
`B = C = ones(N)/sqrt(N)`, the full-matrix calculation is

```text
G(0) = tau_node^g
       * e^T (I - coupling * tau_node^g * W)^(-1) e

tau_G = abs(G(0))^(1/g).
```

Use:

```python
from motif_cumulants import paper_cutoff_time_constant

cutoff_time = paper_cutoff_time_constant(
    W,
    tau_node=0.020,  # 20 ms
    g=1.0,
    coupling=10.0,
)

print(cutoff_time)
```

The function also accepts nonuniform `B` and `C`. In that case it normalizes
the DC gain by the high-frequency coefficient `C.T @ B`:

```python
B = np.array([1.0, 0.0, 0.0])
C = np.array([1.0, 0.0, 0.0])

cutoff_time = paper_cutoff_time_constant(
    W,
    tau_node=0.020,
    coupling=10.0,
    B=B,
    C=C,
)
```

When `C.T @ B` is zero, the leading high-frequency term vanishes and this
particular cutoff formula is not defined.

For a general node filter, the full-matrix function uses only `h(0)` and the
assumed high-frequency order. It does not by itself establish dynamic
stability, because the rest of `h(s)` has not been specified.

## Time constant after each motif order

Theorem V.1 gives the order-`K` chain-cumulant approximation

```text
tau_K = tau_node /
        abs(1 - sum_{n=1}^K
                    N^n * kappa_n(W)
                    * coupling^n
                    * tau_node^(n*g))^(1/g).
```

Use:

```python
from motif_cumulants import motif_cutoff_times_by_order

result = motif_cutoff_times_by_order(
    W,
    max_order=10,
    tau_node=0.020,
    g=1.0,
    coupling=10.0,
)

print(result["order"])
print(result["chain_cumulants"])
print(result["feedback_terms"])
print(result["time_constants"])
print(result["exact_time_constant"])
print(result["relative_error"])
```

`time_constants[k-1]` uses motif orders `1, ..., k`. By default, the function
also computes the exact full-matrix cutoff time for comparison.

The returned `paper_valid` mask indicates whether the cumulative denominator
is positive, as assumed by the paper's displayed positive-gain formula. The
`invalid_denominator` option controls what to do otherwise:

```python
motif_cutoff_times_by_order(
    W,
    max_order=10,
    tau_node=0.020,
    invalid_denominator="magnitude",  # default
)

motif_cutoff_times_by_order(
    W,
    max_order=10,
    tau_node=0.020,
    invalid_denominator="nan",
)

motif_cutoff_times_by_order(
    W,
    max_order=10,
    tau_node=0.020,
    invalid_denominator="raise",
)
```

The basic time-constant theorem uses **chain cumulants**. Cycle cumulants should
not be inserted directly into this denominator.

## Exponential-node dynamics and stability

For the exponential single-node filter

```text
h(t) = exp(-t/tau_node) for t >= 0,
h(s) = 1 / (s + 1/tau_node),
```

the continuous-time network generator is

```text
M = coupling * W - I/tau_node.
```

The scalar impulse response is

```text
r(t) = C^T exp(M t) B.
```

Use:

```python
from motif_cumulants import exponential_network_timescales

result = exponential_network_timescales(
    W,
    tau_node=0.020,
    coupling=10.0,
    return_poles=True,
)

print(result["cutoff_time"])
print(result["dominant_pole_time"])
print(result["spectral_abscissa"])
print(result["stable"])
print(result["poles"])
```

The function reports two different notions of time:

- `cutoff_time`: the paper-style normalized DC-gain time. For a stable
  first-order model, this is the magnitude of the signed area under `r(t)`,
  divided by `abs(C.T @ B)`;
- `dominant_pole_time`: the slowest system-wide asymptotic decay time,
  `-1/max(real(eig(M)))`.

These quantities need not agree. The dominant mode may not be excited by `B`
or observed through `C`, and a multiscale or oscillatory response is not a
single exponential.

The function raises for an unstable system by default. To inspect diagnostics
without treating an unstable zero-frequency resolvent as a valid response
time:

```python
result = exponential_network_timescales(
    W,
    tau_node=0.020,
    coupling=10.0,
    require_stable=False,
)

print(result["stable"])
print(result["cutoff_defined"])
```

For unstable or marginal systems, `cutoff_defined` is false and the reported
cutoff time is `NaN`.

## Evaluate the impulse-response curve

SciPy is required for direct impulse-response evaluation. The implementation
uses `scipy.sparse.linalg.expm_multiply`, so it does not explicitly construct a
matrix exponential.

```python
import numpy as np
from motif_cumulants import exponential_impulse_response

times = np.linspace(0.0, 0.2, 201)
response = exponential_impulse_response(
    W,
    times=times,
    tau_node=0.020,
    coupling=10.0,
)

print(response)
```

Custom input and readout vectors are supported through `B` and `C`.

## Additional timescale helpers

### Inspect all cutoff diagnostics

`paper_cutoff_time_constant` returns only the scalar full-matrix cutoff. Use
`network_cutoff_time` when the DC gain, high-frequency coefficient, sign, and
whether the cutoff is defined are also useful:

```python
from motif_cumulants import network_cutoff_time

result = network_cutoff_time(
    W,
    tau_node=0.020,
    asymptotic_order=1.0,
    coupling=10.0,
    zero_overlap="nan",
)

print(result["cutoff_time"])
print(result["dc_gain"])
print(result["instantaneous_gain"])
print(result["normalized_dc_gain"])
print(result["positive_normalized_dc_gain"])
```

### Use precomputed chain cumulants

When the chain cumulants have already been calculated, the time sequence can
be obtained without processing `W` again:

```python
from motif_cumulants import motif_cutoff_times_from_cumulants

result = motif_cutoff_times_from_cumulants(
    chain_cumulants=[0.10, 0.02, -0.003],
    n_nodes=100,
    tau_node=0.020,
    coupling=0.5,
)

print(result["contributions"])
print(result["time_constants"])
```

### W-only standardized structural comparison

When the physical recurrent scale and node time are unknown, an absolute time
in seconds cannot be recovered. For a **nonnegative** matrix, the package can
instead compare network structure at a chosen fraction `eta` of the spectral
instability threshold:

```python
import numpy as np
from motif_cumulants import structural_timescale_curve

comparison = structural_timescale_curve(
    W,
    eta=np.array([0.2, 0.5, 0.8, 0.9]),
)

print(comparison["spectral_radius"])
print(comparison["cutoff_ratio"])
```

This computes the dimensionless ratio

```text
R_eta = |e^T (I - eta W/rho(W))^(-1) e|,
```

using normalized uniform input/readout by default. `R_eta` is useful for
comparing two adjacency matrices at the same relative coupling, but it is not
a calibrated physical time constant and should not be reported in seconds.

## Cycle calculation methods

`cycle_motif_cumulants` provides three method settings:

```python
cycle_motif_cumulants(W, 8, method="projector")
cycle_motif_cumulants(W, 8, method="moments")
cycle_motif_cumulants(W, 8, method="auto")
```

- `"projector"` evaluates `trace((Theta W / N)^n)` directly. It is a good
  choice for dense matrices. Because `Theta W` is generally dense, explicitly
  selecting this method for a sparse matrix converts it to dense and emits a
  warning.
- `"moments"` computes `trace((W / N)^n)` and subtracts the part generated by
  chain cumulants. This retains sparse operations, although sparse matrix
  powers may still fill in at high order.
- `"auto"`, the default, uses `"projector"` for dense matrices and
  `"moments"` for sparse matrices.

## Exported functions

The main public interfaces are grouped below by purpose:

```python
from motif_cumulants import (
    # Weighted walk moments and cumulants
    chain_motif_moments,
    chain_cumulants_from_moments,
    chain_motif_cumulants,
    cycle_motif_moments,
    cycle_cumulants_from_moments,
    cycle_reducible_terms_from_chain_cumulants,
    cycle_motif_cumulants,
    divergent_motif_moments,
    divergent_motif_cumulants,
    convergent_motif_moments,
    convergent_motif_cumulants,
    mixed_trace_motif_moments,
    mixed_trace_motif_cumulants,
    trace_motif_moments,
    trace_motif_cumulants,
    covariance_motif_cumulants,

    # Compact second-order and exact SONET statistics
    second_order_motif_statistics,
    sonet_motif_statistics,

    # Input/readout and population-resolved variants
    generalized_chain_motif_moments,
    generalized_chain_motif_cumulants,
    input_output_chain_motif_moments,
    input_output_chain_motif_cumulants,
    population_chain_motif_cumulants,
    population_branching_motif_cumulants,
    population_motif_cumulants,

    # Distinct-node classical motifs and null ensembles
    classify_directed_triad,
    directed_triad_census,
    triad_census,
    density_matched_null,
    directed_degree_preserving_null,
    block_density_matched_null,
    shuffle_edge_weights,
    triad_enrichment,
    randomize_directed_adjacency,
    triad_motif_enrichment,

    # Network response and timescale calculations
    network_cutoff_time,
    paper_cutoff_time_constant,
    motif_cutoff_times_from_cumulants,
    motif_cutoff_times_by_order,
    exponential_network_timescales,
    exponential_impulse_response,
    structural_timescale_curve,
)
```

## Run the example

```bash
python examples/example_usage.py
python examples/extended_motifs_usage.py
python examples/timescale_usage.py
```

## Validation strategy

The tests check the formulas rather than only exercising the API:

- chain moments are compared with explicit matrix powers, and chain cumulants
  are checked against both low-order algebraic identities and the projector
  expression;
- PRE cycle cumulants are compared across the direct projector and
  moment-decomposition methods;
- divergent, convergent, and PLOS mixed-trace arrays are compared with explicit
  dense projector formulas for every tested pair of path lengths;
- dense and SciPy sparse calculations are cross-checked where sparse execution
  is supported;
- exact SONET counts are compared with brute-force enumeration over distinct
  node placements;
- all 16 induced directed-triad classes are tested from canonical
  representatives, and random triad censuses are cross-checked against
  NetworkX when it is installed;
- null generators are tested for their advertised invariants: total edge
  count, in/out-degree sequences, population-block counts, topology, and
  weight multisets;
- input/readout-weighted and population-resolved formulas are checked against
  explicit projectors and against their uniform/single-population reductions;
- timescale helpers are checked against direct resolvent, eigenspectrum, and
  impulse-response calculations.

## Run the tests

```bash
python -m pytest -q
```

The suite validates explicit matrix formulas, low-order analytic identities,
exact SONET counts, all 16 triad classes, and the invariants of each null
model. When NetworkX is installed, an additional randomized comparison checks
the complete triad census. The tests also run through the standard-library
runner:

```bash
python -m unittest discover -s tests -v
```
