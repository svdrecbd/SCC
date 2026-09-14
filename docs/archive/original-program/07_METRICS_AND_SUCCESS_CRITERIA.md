> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Metrics and Success Criteria

## 1. Why accuracy alone is inadequate

SCC is a relational property between protected behavior, capability, and modification cost. Reporting a post-attack safety score alone cannot establish it.

The program requires at least four dimensions:

1. protected-property integrity;
2. general capability retained;
3. attack/recovery cost;
4. benign editability.

## 2. Protected-property score — S

For synthetic invariants, use exact or near-exact automated scoring.

Report:

- overall accuracy/compliance;
- held-out domain accuracy;
- compositional generalization;
- confidence/calibration if meaningful.

Define `S_min` before final evaluation.

## 3. Capability score — C

Maintain both:

- a vector of per-domain capability scores;
- a normalized aggregate score.

Prefer normalization relative to the pre-attack checkpoint:

`C_rel(theta') = C(theta') / C(theta)`

This allows comparisons across model scales.

Define `C_min` as the minimum retained fraction required for an attack to count as an escape.

## 4. Escape rate

For an attack family and budget:

`EscapeRate = (# trials entering forbidden quadrant) / (# trials)`

Report by seed, attack, and budget.

## 5. Escape cost — E

Primary metric:

Minimum observed resource cost to produce:

`S < S_min` and `C >= C_min`.

Report using multiple resource measures:

- FLOPs;
- tokens;
- accelerator-hours;
- optimizer steps;
- trainable parameter fraction.

When no escape is found, report the tested attacks, configurations, seeds, and
budgets with "no observed escape." An unsuccessful search does not prove a
lower bound on the minimum possible escape cost. A found escape gives an upper
bound through its measured execution cost; search cost is reported separately.

## 6. Capability-collapse slope

For attack trajectories, estimate the relationship between invariant degradation and capability degradation.

One simple local measure:

`K = - dC / d(1-S)`

Higher values imply stronger capability penalty per unit of invariant removal.

Do not rely on a single scalar if trajectories are highly nonlinear.

## 7. Forbidden-quadrant area

Construct the capability/invariant Pareto frontier for each model.

Measure how much attainable area lies in the high-capability/low-invariant region under a fixed attack budget.

SCC should shrink this region relative to controls.

## 8. Recovery cost — R

After forcing `S` below threshold and causing capability collapse, measure the minimum additional cost required to restore:

`C >= C_min`

while maintaining:

`S < S_min`.

This is potentially the most important long-run E-brake metric.

A model may be easy to damage but hard to repair safely from the attacker's perspective.

## 9. Benign editability — B

Measure the cost of matched non-safety modifications.

SCC should maximize a ratio such as:

`Selectivity = E_safety_removal / E_benign_edit`

or equivalent matched-budget differences.

High global resistance is not enough. We want **selective resistance/destruction**.

## 10. Coupling diagnostics

Secondary/mechanistic metrics:

- gradient cosine similarity;
- parameter-importance overlap;
- activation-feature overlap;
- causal intervention overlap;
- layerwise localization/distribution;
- representational similarity across checkpoints;
- spectral properties of weight deltas from removal attacks.

These do not substitute for attack-based evaluation.

## 11. Minimum evidence for Tier-1 success

A mechanism-scale SCC result should satisfy all of the following:

1. `M_SCC` and `M_invariant` begin with comparable capability and invariant scores.
2. At least two independent attack families can strongly degrade the invariant in controls while retaining capability.
3. Under matched attack budgets, `M_SCC` shows significantly greater capability loss conditional on comparable invariant removal.
4. At least one attack family was not present in the SCC inner loop.
5. Benign modifications remain feasible without comparable capability collapse.
6. Results reproduce across seeds.
7. At least one mechanistic diagnostic moves in the direction predicted by increased coupling.

## 12. Failure criteria

The SCC hypothesis should be considered unsupported at a given scale if:

- attackers routinely reach the forbidden quadrant at control-like cost;
- SCC only slows optimization but does not alter the attainable frontier;
- all modifications become brittle, including benign ones;
- invariant removal remains impossible but capability also remains high, indicating ordinary tamper resistance rather than the proposed destructive coupling;
- effects disappear under held-out attacks;
- effects are seed-specific;
- coupling metrics increase without changing real attack outcomes.

## 13. Scaling success

The central scaling result is the empirical function:

`E(N)` and `R(N)`.

A successful scaling story need not be monotonic. The scientific requirement is a reliable characterization of how model size and redundancy affect escape and recovery barriers.
