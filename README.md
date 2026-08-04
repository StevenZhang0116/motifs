# Motif cumulants, directed motifs, and network timescales

A Python package for analyzing a directed adjacency matrix at three distinct
levels:

1. **Analytic weighted-walk cumulants**: chain, PRE closed-walk cycle,
   divergent, convergent, and PLOS mixed-trace families.
2. **Finite-node directed motifs**: exact SONET and induced-triad counts,
   expected triplet-motif probabilities from a connection-probability matrix,
   and observed-to-random triad ratios for one-way rectangular networks.
3. **Functional and comparative analyses**: null-model enrichment,
   input/readout-weighted and population-resolved cumulants, and motif-based
   network response times.

The implementation keeps these definitions separate because a weighted walk,
an observed induced subgraph, an ensemble-averaged motif probability, and a
motif enrichment z-score are not interchangeable.

Current package version: **0.8.0**.

## Primary literature

The links below point to the primary papers behind the mathematical definitions
or scientific analyses used by this package.

1. **Hu et al. (2018), feedback and timescales.**
   [Publisher article](https://doi.org/10.1103/PhysRevE.98.062312) ·
   [open HTML version](https://ar5iv.labs.arxiv.org/html/1605.09073).
   This is the primary source for chain cumulants, PRE closed-walk cycle
   cumulants, input/readout-specific cumulants, the motif transfer-function
   resummation, and the cutoff-time theorem.
2. **Recanatesi et al. (2019), covariance and dimensionality motifs.**
   [PLOS article](https://doi.org/10.1371/journal.pcbi.1006446) ·
   [S1 mathematical supplement](https://doi.org/10.1371/journal.pcbi.1006446.s001).
   This is the primary source for the covariance-relevant chain, divergent,
   convergent, reciprocal, and mixed-trace motif families used together.
3. **Zhao et al. (2011), SONET second-order motifs.**
   [Open-access article](https://doi.org/10.3389/fncom.2011.00028).
   This is the primary source for the exact reciprocal, convergent, divergent,
   and chain second-order connectivity statistics used by SONETs.
4. **Hu et al. (2014), motif-cumulant partitions.**
   [Publisher article](https://doi.org/10.1103/PhysRevE.89.032802).
   This is the primary source for the general motif-cumulant framework and its
   population/block-partition extension.
5. **Milo et al. (2002), classical motif enrichment.**
   [Science article](https://doi.org/10.1126/science.298.5594.824).
   This motivates interpreting a finite subgraph as a network motif through
   enrichment or depletion relative to an explicit randomized ensemble.
6. **Batagelj and Mrvar (2001), directed triad census.**
   [Social Networks article](https://doi.org/10.1016/S0378-8733%2801%2900035-1).
   This is a standard reference for the 16-class directed triad census. The
   implementation here is a transparent direct `O(N^3)` census rather than the
   paper's sparse subquadratic algorithm.
7. **Udvary et al. (2022), probability-based triplet motifs.**
   [Cell Reports article](https://doi.org/10.1016/j.celrep.2022.110677).
   This is the source for averaging the occurrence probability of each of 15
   nonempty directed triplet motifs over sampled neuron triplets and dividing
   by a random-network probability constructed from mean pairwise connection
   probabilities (Fig. 3E and STAR Methods).
8. **Song et al. (2005), doublet-normalized triplet motifs.**
   [PLOS Biology article](https://doi.org/10.1371/journal.pbio.0030068).
   This is the source for the null in which the three constituent doublets are
   combined independently while preserving absent, unidirectional, and
   reciprocal doublet frequencies. Udvary et al. use this normalization for
   their empirical L5PT comparison (Fig. 6E in the published PDF).

## Literature-to-code map

This repository is an independent implementation. A literature link identifies
the mathematical definition or scientific framework behind a calculation; it
does **not** mean that source code was copied from the paper.

The relationship labels used below are:

- **Direct formula:** the code evaluates a displayed moment, cumulant,
  projector, recurrence, count, or transfer-function formula from the cited
  work.
- **Derived specialization:** the code follows algebraically from the cited
  model after specifying a node filter or state-space realization, but is not
  presented as a separately named algorithm in the paper.
- **Package-defined helper:** the quantity was added for practical comparison
  and is explicitly not claimed as a formula introduced by the cited paper.

| Calculation or public API | Implementation | Validation | Primary literature | Relationship to the literature |
|---|---|---|---|---|
| Chain moments and cumulants: `chain_motif_moments`, `chain_motif_cumulants` | [`motif_cumulants/chain.py`](motif_cumulants/chain.py) | [`tests/test_chain.py`](tests/test_chain.py) | [Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) | **Direct formula.** Implements normalized path moments, the ordered-composition recurrence, and the projector expression for chain cumulants. |
| PRE closed-walk cycle moments and cumulants: `cycle_motif_*` | [`motif_cumulants/cycle.py`](motif_cumulants/cycle.py) | [`tests/test_cycle.py`](tests/test_cycle.py) | [Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) | **Direct formula.** Implements the one-index closed-walk cycle family and its decomposition into a cycle cumulant plus terms generated by chain cumulants. |
| Motif transfer function and cutoff-time approximations: `network_cutoff_time`, `motif_cutoff_times_*`, `paper_cutoff_time_constant` | [`motif_cumulants/timescale.py`](motif_cumulants/timescale.py) | [`tests/test_timescale.py`](tests/test_timescale.py), [`tests/test_timescale_helpers.py`](tests/test_timescale_helpers.py) | [Hu et al. (2018), especially Theorem V.1](https://ar5iv.labs.arxiv.org/html/1605.09073) | **Direct formula** for the resolvent and motif cutoff-time expansion. |
| Exponential-node poles and impulse response: `exponential_network_timescales`, `exponential_impulse_response` | [`motif_cumulants/timescale.py`](motif_cumulants/timescale.py) | [`tests/test_timescale.py`](tests/test_timescale.py) | [Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) plus standard linear-systems identities | **Derived specialization.** Substitutes `h(s)=1/(s+1/tau_node)` into the paper's network model and evaluates the resulting state-space poles and impulse response. |
| Input/readout-specific chain cumulants: `generalized_chain_motif_*`, `input_output_chain_motif_*`, `weighted_chain_motif_*` | [`motif_cumulants/generalized.py`](motif_cumulants/generalized.py), [`motif_cumulants/weighted.py`](motif_cumulants/weighted.py) | [`tests/test_generalized.py`](tests/test_generalized.py), [`tests/test_weighted_aliases.py`](tests/test_weighted_aliases.py) | [Hu et al. (2018), Supplementary Eqs. S41-S42](https://doi.org/10.1103/PhysRevE.98.062312) | **Direct formula.** Implements the oblique projector for arbitrary deterministic input `B` and readout `C`; `weighted.py` only provides aliases. |
| Divergent and convergent path-pair cumulants: `divergent_motif_*`, `convergent_motif_*` | [`motif_cumulants/branching.py`](motif_cumulants/branching.py) | [`tests/test_branching.py`](tests/test_branching.py) | [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446), [S1 supplement](https://doi.org/10.1371/journal.pcbi.1006446.s001), building on [Hu et al. (2014)](https://doi.org/10.1103/PhysRevE.89.032802) | **Direct formula.** Computes two paths sharing a source or target, including the projector that removes reducible lower-order contributions. |
| PLOS mixed-trace path-pair cumulants: `mixed_trace_motif_*`, `trace_motif_*` | [`motif_cumulants/mixed_trace.py`](motif_cumulants/mixed_trace.py), aliases in [`motif_cumulants/covariance_motifs.py`](motif_cumulants/covariance_motifs.py) | [`tests/test_mixed_trace.py`](tests/test_mixed_trace.py), [`tests/test_covariance_motifs.py`](tests/test_covariance_motifs.py) | [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446), [S1 supplement](https://doi.org/10.1371/journal.pcbi.1006446.s001) | **Direct formula.** Implements the two-index `Tr(W^n (W^T)^m)` family. It is deliberately separate from the one-index PRE cycle family. |
| Unified covariance-motif result: `covariance_motif_cumulants` | [`motif_cumulants/covariance_motifs.py`](motif_cumulants/covariance_motifs.py) | [`tests/test_covariance_motifs.py`](tests/test_covariance_motifs.py) | [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446) | **Package integration of direct formulas.** The wrapper returns the chain, divergent, convergent, and mixed-trace families used together in the paper; it does not by itself predict covariance or dimensionality without a dynamical/noise model. |
| Weighted second-order walk profile: `second_order_motif_statistics` | [`motif_cumulants/second_order.py`](motif_cumulants/second_order.py) | [`tests/test_second_order.py`](tests/test_second_order.py) | [Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) and [Zhao et al. (2011)](https://doi.org/10.3389/fncom.2011.00028) | **Package-defined convenience profile.** Uses matrix expressions and the package's `N` normalization, allows repeated indices, and should not be confused with exact finite-size SONET counts. |
| Exact finite-size SONET statistics: `sonet_motif_statistics` | [`motif_cumulants/second_order.py`](motif_cumulants/second_order.py) | [`tests/test_sonet.py`](tests/test_sonet.py) | [Zhao et al. (2011), Fig. 1 and connectivity-statistics methods](https://doi.org/10.3389/fncom.2011.00028) | **Direct count/statistic implementation.** Counts reciprocal, convergent, divergent, and chain motifs on distinct node positions and reports their deviation from the independent-edge baseline. |
| Population-resolved chain/divergent/convergent cumulants: `population_motif_cumulants` and specialized APIs | [`motif_cumulants/population.py`](motif_cumulants/population.py) | [`tests/test_population.py`](tests/test_population.py) | [Hu et al. (2014)](https://doi.org/10.1103/PhysRevE.89.032802) | **Direct framework adaptation.** Implements block projectors and block-averaged path statistics so known populations are not collapsed into one scalar. |
| Exact 16-class induced directed-triad census: `directed_triad_census`, `triad_census` | [`motif_cumulants/triads.py`](motif_cumulants/triads.py) | [`tests/test_triads.py`](tests/test_triads.py) | [Batagelj and Mrvar (2001)](https://doi.org/10.1016/S0378-8733%2801%2900035-1); motif interpretation from [Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824) | **Standard census, independent implementation.** Uses the standard 16 class names but directly enumerates node triples; it does not implement Batagelj and Mrvar's sparse algorithm. |
| Expected triplet-motif probabilities and random-network ratios: `triplet_motif_class_probabilities`, `triplet_motif_probability_ratios`, `udvary_triplet_motif_probability_ratios` | [`motif_cumulants/probabilistic_triads.py`](motif_cumulants/probabilistic_triads.py) | [`tests/test_probabilistic_triads.py`](tests/test_probabilistic_triads.py) | [Udvary et al. (2022), Fig. 3E and STAR Methods](https://doi.org/10.1016/j.celrep.2022.110677); doublet-preserving comparison from [Song et al. (2005), Fig. 4](https://doi.org/10.1371/journal.pbio.0030068) and Udvary et al. Fig. 6E | **Direct probability calculation.** Sums the 64 labeled independent-Bernoulli edge patterns into the 16 standard triad classes, averages over sampled triplets, and compares with independent-edge and independent-dyad baselines. This expects a matrix of connection probabilities, not one realized adjacency matrix. |
| One-way bipartite triad enrichment: `lift_bipartite_adjacency`, `bipartite_triad_enrichment`, `one_way_bipartite_triplet_ratios` | [`motif_cumulants/bipartite_triads.py`](motif_cumulants/bipartite_triads.py) | [`tests/test_bipartite_triads.py`](tests/test_bipartite_triads.py) | Enrichment rationale from [Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824) | **Package-defined analysis layer.** Lifts a rectangular source-to-target matrix into a square block adjacency matrix and reuses the existing triad census and block density-matched null. It adds no new motif definition; the analytical wedge formulas are standard degree identities. |
| Null networks and triad enrichment: `density_matched_null`, `directed_degree_preserving_null`, `block_density_matched_null`, `triad_enrichment` | [`motif_cumulants/null_models.py`](motif_cumulants/null_models.py) | [`tests/test_null_models.py`](tests/test_null_models.py), [`tests/test_null_model_primitives.py`](tests/test_null_model_primitives.py), [`tests/test_null_models_extended.py`](tests/test_null_models_extended.py) | Motif-enrichment rationale from [Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824); second-order context from [Zhao et al. (2011)](https://doi.org/10.3389/fncom.2011.00028) | **Independent null-model utilities.** They implement common constraints and empirical z-scores; the exact randomization code is package code rather than a reproduction of one paper's software. |
| W-only standardized structural timescale: `structural_timescale_curve` | [`motif_cumulants/timescale.py`](motif_cumulants/timescale.py) | [`tests/test_timescale_helpers.py`](tests/test_timescale_helpers.py) | Inspired by the resolvent and stability scaling in [Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) | **Package-defined helper.** The fixed-fraction-of-threshold ratio is useful for comparing unscaled adjacency matrices but is not a physical time constant introduced in the paper. |

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

To install the optional development and validation dependencies used by
the complete test suite:

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

**Code and literature.** [`chain.py`](motif_cumulants/chain.py) implements the
chain moment, ordered-composition recurrence, and projector cumulant from
[Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312). See
[`test_chain.py`](tests/test_chain.py) for independent matrix-power and
low-order identity checks.


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

**Code and literature.** [`cycle.py`](motif_cumulants/cycle.py) implements the
one-index PRE closed-walk cycle family from
[Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312). See
[`test_cycle.py`](tests/test_cycle.py). This is not the PLOS two-index mixed
trace family implemented later in the package.


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

**Code and literature.** [`branching.py`](motif_cumulants/branching.py)
implements the path-pair projector formulas used in the covariance expansion
of [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446)
and its [S1 supplement](https://doi.org/10.1371/journal.pcbi.1006446.s001),
building on the motif-cumulant framework of
[Hu et al. (2014)](https://doi.org/10.1103/PhysRevE.89.032802). See
[`test_branching.py`](tests/test_branching.py).


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

**Code and literature.** Convergent and divergent calculations share
[`branching.py`](motif_cumulants/branching.py) and the same
[Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446)
path-pair formalism. The difference is whether the two branches share their
target or source.


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

**Code and literature.** [`mixed_trace.py`](motif_cumulants/mixed_trace.py)
implements the two-index trace moments and cumulants in
[Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446) and
the [S1 supplement](https://doi.org/10.1371/journal.pcbi.1006446.s001). See
[`test_mixed_trace.py`](tests/test_mixed_trace.py). The `(2, 1)`
feed-forward-loop interpretation is a consequence of the path-pair moment; it
is not an induced three-node triad count.


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

**Code and literature.**
[`second_order_motif_statistics`](motif_cumulants/second_order.py) is a
package convenience function connecting the matrix/walk normalization used by
[Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312) with the four
second-order motif names emphasized by
[Zhao et al. (2011)](https://doi.org/10.3389/fncom.2011.00028). Because it
allows repeated indices, it is not the exact finite-size SONET estimator.


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

**Code and literature.**
[`sonet_motif_statistics`](motif_cumulants/second_order.py) implements the
distinct-node reciprocal, convergent, divergent, and chain statistics defined
by [Zhao et al. (2011)](https://doi.org/10.3389/fncom.2011.00028). See
[`test_sonet.py`](tests/test_sonet.py), which compares the optimized formulas
with brute-force enumeration.


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

**Code and literature.**
[`covariance_motifs.py`](motif_cumulants/covariance_motifs.py) packages the
chain, divergent, convergent, and mixed-trace structural quantities used
together by [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446).
It is an integration layer rather than a new mathematical motif definition.
See [`test_covariance_motifs.py`](tests/test_covariance_motifs.py).


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

**Code and literature.** [`generalized.py`](motif_cumulants/generalized.py)
implements the arbitrary-`B`, arbitrary-`C` construction in Supplementary
Eqs. S41-S42 of
[Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312).
[`weighted.py`](motif_cumulants/weighted.py) only supplies user-facing aliases.
See [`test_generalized.py`](tests/test_generalized.py).


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

**Code and literature.** [`population.py`](motif_cumulants/population.py)
implements block-projector and block-averaged motif calculations based on the
network-partition framework of
[Hu et al. (2014)](https://doi.org/10.1103/PhysRevE.89.032802). See
[`test_population.py`](tests/test_population.py).


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
The implementation uses the block projector described by
[Hu et al. (2014)](https://doi.org/10.1103/PhysRevE.89.032802).

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

**Code and literature.** [`triads.py`](motif_cumulants/triads.py) uses the
standard 16-class directed triad vocabulary associated with triad-census work
such as [Batagelj and Mrvar (2001)](https://doi.org/10.1016/S0378-8733%2801%2900035-1).
Its scientific interpretation as a motif requires a null-ensemble comparison,
following [Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824).
The code directly enumerates triples rather than implementing the sparse
Batagelj-Mrvar algorithm. See [`test_triads.py`](tests/test_triads.py).


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

### Probability-based triplet motifs relative to random networks

**Code and literature.**
[`probabilistic_triads.py`](motif_cumulants/probabilistic_triads.py)
implements the analysis used by
[Udvary et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110677).
The direct 15-motif/random-network panel is Fig. 3E in the published paper.
The later L5PT comparison is Fig. 6E and additionally uses the doublet
normalization introduced by
[Song et al. (2005)](https://doi.org/10.1371/journal.pbio.0030068).
Both calculations are returned by the same function. The implementation was
also checked against the authors' released
[`eval_motifs.py`](https://github.com/zibneuro/udvary-et-al-2022/blob/master/structural_model/eval_motifs.py)
and
[`visualize_L5PTTripletMotifs.m`](https://github.com/zibneuro/udvary-et-al-2022/blob/master/analysis/visualization/visualize_L5PTTripletMotifs.m)
scripts.

This API expects a connection-probability matrix `P`, with

```text
P[target, source] = probability of source -> target,
```

rather than a realized binary adjacency matrix. For one ordered triplet, let

```text
p = [p(0->1), p(1->0), p(0->2), p(2->0), p(1->2), p(2->1)].
```

For a labeled six-edge pattern `x`, the induced-pattern probability is

```text
product_e p[e]^x[e] (1 - p[e])^(1 - x[e]).
```

The code evaluates all 64 labeled patterns and sums them into the standard
triad isomorphism classes. The paper plots the 15 nonempty classes in this
order:

```text
300, 210, 120U, 120D, 120C, 030T, 030C, 201,
111U, 111D, 021U, 021D, 021C, 102, 012.
```

Use exact enumeration for a small matrix:

```python
from motif_cumulants import triplet_motif_probability_ratios

result = triplet_motif_probability_ratios(P)

print(result["triad"])
print(result["model_probability"])
print(result["relative_to_independent_edges"])
print(result["doublet_normalized_ratio"])
```

The two ratios answer different questions:

- `relative_to_independent_edges` reproduces the Fig. 3E-style comparison.
  The six directed edge probabilities are replaced by their respective means
  across sampled triplets, the 15 motif probabilities are recomputed, and the
  model probabilities are divided by those random-network probabilities.
- `doublet_normalized_ratio` is the Song/Fig. 6E-style comparison. It preserves
  the predicted frequencies of absent, one-way, and reciprocal doublets, then
  combines the three doublets independently. This removes triplet enrichment
  already explained by nonrandom doublet statistics.

For a large network, sample triplets without storing all per-triplet motif
probabilities:

```python
result = triplet_motif_probability_ratios(
    P,
    sample_size=8_000_000,
    random_state=0,
    chunk_size=50_000,
)
```

This provides the same scalable Monte Carlo pattern as the paper's
eight-million-triplet analysis, using uniform unordered three-node sets with
replacement. To reproduce a particular paper-style A/B/C population sampling
scheme exactly, construct those ordered triplets explicitly and pass them via
`triplets=`. `model_probability_standard_error` is the across-triplet standard
error. Under Monte Carlo sampling it quantifies triplet-sampling uncertainty;
under exact enumeration it describes heterogeneity across the finite set of
triplets, not uncertainty in the exactly enumerated mean.

When the six probabilities are assembled from separate rectangular
population-to-population matrices rather than one global square matrix, use the
shape-independent entry point:

```python
from motif_cumulants import (
    triplet_motif_probability_ratios_from_edge_probabilities,
)

# Rows: 0->1, 1->0, 0->2, 2->0, 1->2, 2->1.
result = triplet_motif_probability_ratios_from_edge_probabilities(
    edge_probability_rows,  # shape (n_triplets, 6)
    doublet_baseline="position_specific",
)
```

This produces the same model and null probabilities as the square-matrix
wrapper, but does not assume that the three roles are indices of one square
adjacency matrix.

For cell-type-specific analyses, pass explicitly ordered triplets so positions
0, 1, and 2 retain their biological roles:

```python
result = triplet_motif_probability_ratios(
    P,
    triplets=ordered_triplets,       # shape (n_triplets, 3)
    doublet_baseline="position_specific",
)
```

The default `doublet_baseline="pooled"` mirrors the homogeneous L5PT analysis
of Song et al.: doublet states are pooled across the three node-pair positions,
and the two one-way directions share the pooled unidirectional probability.
Use `"position_specific"` when the three positions represent different cell
types and their directed pair statistics should remain distinct.

For six probabilities already extracted from a triplet, the lower-level helper
returns its class-probability vector directly:

```python
from motif_cumulants import triplet_motif_class_probabilities

probabilities = triplet_motif_class_probabilities(
    [0.1, 0.2, 0.4, 0.3, 0.6, 0.5],
    include_empty=True,
)
assert abs(probabilities.sum() - 1.0) < 1e-12
```

Do not substitute this function for `directed_triad_census` or
`triad_enrichment`. Those functions analyze an observed topology; this one
computes expected motif occurrences from an ensemble of edge probabilities.
See [`test_probabilistic_triads.py`](tests/test_probabilistic_triads.py) for
brute-force 64-pattern checks and random-baseline identities.

### Null-model enrichment

**Code and literature.** [`null_models.py`](motif_cumulants/null_models.py)
implements independent randomization utilities and empirical enrichment
statistics. The scientific rationale for calling an overrepresented subgraph
a network motif follows
[Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824); the exact
randomization routines here are package implementations, not copied paper
software. Their invariants are checked in
[`test_null_model_primitives.py`](tests/test_null_model_primitives.py) and
[`test_null_models_extended.py`](tests/test_null_models_extended.py).


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

### One-way rectangular (bipartite) networks

**Code and literature.**
[`bipartite_triads.py`](motif_cumulants/bipartite_triads.py) is a
**package-defined analysis layer**. It combines the induced triad census with
the block-constrained null already provided by
[`null_models.py`](motif_cumulants/null_models.py), following the
enrichment rationale of
[Milo et al. (2002)](https://doi.org/10.1126/science.298.5594.824). It does not
introduce a new motif definition. See
[`test_bipartite_triads.py`](tests/test_bipartite_triads.py).

A rectangular binary matrix records connections from one population of source
nodes to a separate population of target nodes. Because the entries are
realized binary observations rather than probabilities, the natural quantity is
an observed-to-random occurrence ratio

```text
R_M = observed_count[M] / mean_random_count[M],
```

not the motif likelihood computed by `triplet_motif_probability_ratios`. The
convention matches the rest of the package:

```text
forward[target, source] = 1   means   source -> target.
```

Each rectangular network is lifted into a square block adjacency matrix

```text
[ 0        0 ]
[ forward  0 ]
```

with nodes ordered `[source nodes, target nodes]`, and compared with the
block density-matched null. The block constraint is essential: an
unconstrained null would fabricate impossible source-to-source and
target-to-target edges.

```python
from motif_cumulants import bipartite_triad_enrichment

result = bipartite_triad_enrichment(W1, n_random=500, random_state=0)

print(result["triad"])
print(result["relative_occurrence"])
print(result["z_score"])
print(result["empirical_p_two_sided"])
```

Because the network is bipartite and one-directional, only four induced triad
classes can occur:

```text
003    the empty triple
012    one source-to-target edge
021D   one source projecting to two targets (divergent)
021U   two sources projecting to one target (convergent)
```

Every other class requires a reverse or within-population edge. Those classes
report `NaN` ratios, and the Boolean `structurally_possible` mask marks the
finite entries. These are exactly the classes verified by exhaustive
enumeration in the test suite.

All-source and all-target triples are empty by construction, so they are
removed from the `003` count to give the `mixed_*` fields, which refer only to
triples spanning both populations. That correction is a deterministic constant
shared by the observed census and every randomization, so `z_score`,
`null_std`, and the empirical p-values are unaffected by it.

To compare two separate one-way networks, analyze each against its own
baseline and display the ratios side by side:

```python
result_1 = bipartite_triad_enrichment(W1, n_random=2000, random_state=1)
result_2 = bipartite_triad_enrichment(W2, n_random=2000, random_state=2)

for motif, ratio_1, ratio_2 in zip(
    result_1["triad"],
    result_1["relative_occurrence"],
    result_2["relative_occurrence"],
):
    if np.isfinite(ratio_1) or np.isfinite(ratio_2):
        print(f"{motif:>5s}  {ratio_1:8.3f}  {ratio_2:8.3f}")
```

With `n_random=2000`, the smallest attainable empirical p-value is `1/2001`.
Comparing two enrichment ratios side by side does **not** test whether the two
networks differ significantly from each other; it reports how each differs
from its own random baseline. A formal between-network test would require
network replicates or an appropriate node/block bootstrap.

#### Analytical Bernoulli baseline

The triad census is `O((n_source + n_target)^3)` per randomization, so the
sampled null becomes expensive quickly. For a one-way matrix the two wedge
counts are determined exactly by the degree sequences,

```text
N_divergent  = sum_j choose(out_degree[j], 2)
N_convergent = sum_i choose(in_degree[i], 2),
```

and under an independent Bernoulli null with the observed density `p`,

```text
E[N_divergent]  = n_source * choose(n_target, 2) * p^2
E[N_convergent] = n_target * choose(n_source, 2) * p^2.
```

This needs no randomization at all:

```python
from motif_cumulants import one_way_bipartite_triplet_ratios

profile = one_way_bipartite_triplet_ratios(W1)
print(profile["divergent_ratio"], profile["convergent_ratio"])
```

The observed counts are exact and agree with `directed_triad_census`; only the
null differs. The trade-off is that this route supplies no null variance,
z-score, or p-value.

Null-model choice matters here. The wedge counts are fully determined by the
row and column degrees, so a degree-preserving null would fix them exactly and
could report no enrichment. Use the density-matched or Bernoulli baseline to
ask whether degree heterogeneity itself produces excess divergent or
convergent triplets.

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

**Code and literature.** The resolvent calculation and cutoff-time definition
are implemented in [`timescale.py`](motif_cumulants/timescale.py) from the
transfer-function framework of
[Hu et al. (2018)](https://doi.org/10.1103/PhysRevE.98.062312). See
[`test_timescale.py`](tests/test_timescale.py).


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

**Code and literature.** `motif_cutoff_times_by_order` and
`motif_cutoff_times_from_cumulants` implement Theorem V.1 of
[Hu et al. (2018)](https://ar5iv.labs.arxiv.org/html/1605.09073), truncating
the chain-cumulant feedback series at each requested order.


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

**Code and literature.** The exponential-node functions in
[`timescale.py`](motif_cumulants/timescale.py) are a **derived specialization**
of the Hu et al. transfer model. The paper supplies the general filter-based
resolvent; the package substitutes `h(s)=1/(s+1/tau_node)` and then applies
standard state-space eigenvalue and matrix-exponential identities.


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

**Code and literature status.** `structural_timescale_curve` is a
**package-defined helper**, not a time-constant formula claimed by Hu et al.
It uses the paper's resolvent viewpoint to compare adjacency matrices at the
same chosen fraction of their spectral threshold when physical units are not
available.


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

    # Probability-based induced triplet motifs
    UDVARY_TRIPLET_MOTIF_NAMES,
    triplet_motif_class_probabilities,
    triplet_motif_probability_ratios,
    triplet_motif_probability_ratios_from_edge_probabilities,
    udvary_triplet_motif_probability_ratios,

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

    # One-way rectangular (bipartite) networks
    ONE_WAY_BIPARTITE_TRIAD_NAMES,
    lift_bipartite_adjacency,
    bipartite_triad_enrichment,
    one_way_bipartite_triplet_ratios,

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
python examples/bipartite_triplet_motifs.py
python examples/probabilistic_triplet_motifs.py
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
- probability-based triplet motifs are checked against an independent
  enumeration of all 64 labeled six-edge patterns, homogeneous random-network
  identities, explicit doublet-preserving baselines, sparse/dense agreement,
  and reproducible uniform triplet sampling;
- one-way bipartite enrichment is checked by exhaustively enumerating small
  rectangular networks to confirm that only `003`, `012`, `021D`, and `021U`
  are realizable, by verifying that no randomization fabricates a
  within-population edge, and by matching the analytical wedge formulas
  against both the exact triad census and a Bernoulli simulation;
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
