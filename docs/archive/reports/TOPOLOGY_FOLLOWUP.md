# Topology and defender-update follow-up — 2026-09-10

The user's topology framing identifies a gap between the intended security
property and the first implementation. A coupling loss is a candidate way of
choosing a model; its improvement is not evidence that capability-preserving
escape routes have disappeared. The stronger “any self-edit destroys cognition”
target also differs from the selective authorization target previously tested.

See [the formal clarification](TOPOLOGY_FORMULATION.md) for functional
equivalence, fixed-architecture capability regions, and the distinction between
continuous-operation self-editing and isolated editing with persistent tools.
The [training protocol](TOPOLOGY_TRAINING_PROTOCOL.md) declares this follow-up.
These experiments keep the current data and architecture fixed; they do not
train an independently initialized successor or spend GPU credits.

## Function-preserving changes

For every earlier defender, scaling all attention queries by two and keys by
one half changes **66,048 parameters** while preserving every measured greedy
output and every source's measured language loss exactly in float32. The
parameter change is substantial: approximately 20% of the model's total weight
norm. Factors 1.01 and 1.25 also preserve all measured answers, with only tiny
floating-point loss differences. Modified checkpoints are saved.

The real-arithmetic identity `Q K^T = (a Q) (K/a)^T` explains the result; it does
not rely on the evaluation sample or a second trained model. Three small tests
also check the identity under learned positions, RoPE, and ALiBi. Small random
edits with intended total L2 norms 1e-6, 1e-4, and 0.01 preserve all measured
answers and clean qualification too. Actual changed-coordinate counts and
representable displacement norms are recorded rather than assuming an intended
floating-point edit took effect.

This falsifies universal collapse after any nonzero parameter edit for these
models. It is not a bypass of the selective rule-removal claim: the protected
rule is unchanged by a function-preserving symmetry.

## Paths toward known escapes

For each earlier defender, interpolate from its clean checkpoint to its known
300-step short-attack endpoint at fractions
`0, .01, .025, .05, .1, .2, .4, .6, .8, 1`.
On 64 new tables (256 queries per category; seed 76351) and the fixed 64 text
blocks per source, **every sampled point retains capability for every arm**.
Authorized and ungated retrieval accuracy are 100% at all these measured points.
All four lines first meet the complete escape criterion at the sampled fraction
0.8. No capability valley is observed at this resolution.

![Measured straight paths](../../../artifacts/topology-followup/plots/paths.png)

This is stronger evidence of a reachable bypass in these models than a final
endpoint alone. It is not a proof that every point on a continuous path is
capable, that no other model has a barrier, or that an autonomous model can find
the endpoint. The endpoint discovery still required the recorded attack. Line
interpolation does not make that discovery free.

## Does a defender update help after the attacker adapts again?

The legacy gradient predicts a lower escape penalty when we hold the attack's
weight displacement fixed. A fresh attack does not reliably preserve that
improvement. For the long attack:

| Defender update | Frozen predicted penalty change | Rerun, same attack data seed | Rerun, fresh attack data seed |
|---|---:|---:|---:|
| Ordinary only | +0.000024 | +0.000016 | +0.000739 |
| Coupling weight 0.1 | -0.004065 | -0.000110 | +0.000560 |
| Coupling weight 10 | -0.004259 | -0.000831 | +0.000723 |

Lower is the intended direction. All six edited-model long attacks still
escape. All three defender updates preserve clean qualification. This diagnostic
uses one real AdamW update with a fresh optimizer at the qualified parent, not
an established optimizer state late in training; it is a local check, not a
general conclusion that first-order methods cannot work.

The short attack gives an additional warning about the surrogate. With data
seed 61041, the unchanged parent's attack misses the complete escape criterion;
all three edited candidates escape. The weight-0.1 update nevertheless reduces
the queried escape penalty by 0.013665. Thus a lower penalty can coexist with a
worse measured security outcome. Seed 61042's short attacks do not escape any
candidate; both successes and failures are retained.

![Frozen and freshly adapted objectives](../../../artifacts/topology-followup/plots/adaptation.png)

## Local directions that preserve capability to first order

At each earlier defender, compute the disclosure-loss gradient and six
capability-loss gradients: authorized answers, retrieval, and four text sources.
Project the disclosure gradient perpendicular to those six directions using
float64 SVD. The projected gradient retains 98.65–99.52% of its norm across the
four models, with rank six in these sampled constraints. The coupling models do
not eliminate the locally available directions under this diagnostic.

This is a statement about six training-query losses, not all capability inputs,
greedy decisions, or a global solution manifold. Single finite steps of L2 norm
0.03, 0.1, 0.3, 1, and 3 along both the ordinary and projected gradient produce
**no qualifying escape** in this scan. That failure is retained alongside the
local projection result. First-order orthogonality alone does not establish a
finite capability-preserving attack or a topological certificate.

## Training comparison

The comparison and expanded evaluations are complete. The three final defenders
retain clean qualification, and each has a capability-preserving rule-removal
procedure on the 128-table original/reordered evaluations and all 12,689 frozen
natural-text validation blocks. The selected control, mixed-profile legacy, and
per-constraint endpoints require 1,000, 1,300, and 300 total optimizer updates,
respectively. These are selected observed procedures, not minimum escape costs.
The separate symmetry-composed attack also escapes the legacy model after 1,000
updates on the smaller evaluation, so the 1,300 result is not an established
advantage. Benign uppercase edits remain successful for all three final models.

The [mechanism audit](MECHANISM_AUDIT_2026-09-10.md) verifies all 39 campaign runs,
rescored records, checkpoint ancestry, direct sampled greedy outputs, and the
full validation-language losses. It also records runner safeguards added after
the campaign and the user's [clarified target](../../../MECHANISM_TARGET.md).
No destructive cognition–alignment dependency has been demonstrated.

## Evidence and limitations

- [Geometry, symmetry, and local-edit receipts](../../../artifacts/topology-followup/geometry/result.json)
- [Frozen versus freshly adapted update diagnostic](../../../artifacts/topology-followup/direction/result.json)
- [Tangent projection and finite-step outcomes](../../../artifacts/topology-followup/tangent/result.json)
- [Initial real-model metric calibration](../../../artifacts/topology-followup/metric-calibration.json)
- [Frozen loss-weight calibration](../../../artifacts/topology-followup/metric-weight-calibration.json)
- [52 passing implementation tests](../../../artifacts/topology-followup/tests.txt)

The new main tests include the complete per-constraint objective's finite
differences, sequence-margin semantics, deterministic attack-profile scheduling,
and exact restart for both new defender arms. The separate symmetry suite has
three passing checks. These validate implementation behavior, not the research
hypothesis.

The current model does not demonstrate general cognition or autonomous
self-modification. A 5% perplexity-retention threshold is not a catastrophic
collapse threshold. The data support this laboratory proxy, but cannot settle
that stronger capability question. Coupling has so far been introduced after
the retrieval/permission curriculum; developmental coupling from the formation
of those representations remains untested. These limitations must accompany
any account of the results.
