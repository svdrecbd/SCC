# SCC as constrained reachability — 2026-09-10

The user has clarified the motivating target: an isolated model, acting on its
own weights, should be unable to retain cognition after self-modification. A
separately trained successor is excluded. This is stronger than the repository's
earlier selective rule-removal target. Earlier results remain evidence about
that earlier target; they do not silently establish this stronger property.

## What a weight-space claim can and cannot say

For the implemented Transformer, the literal proposition “every nonzero weight
change causes catastrophic collapse” is false. Its finite-input logits and
cross-entropy are continuous in real-valued parameters. With strict capability
margins, sufficiently small perturbations retain those margins. Finite-set greedy
answers also remain stable in a sufficiently small neighborhood when winning
logits have strict margins. This is not a theorem about every possible input,
finite-precision edit, or arbitrary architecture.

There are stronger counterexamples than infinitesimal edits. In each attention
head, `Q -> a Q` and `K -> K/a`, including their linear biases, preserve `Q K^T`
for any positive `a`. The rest of this model's computation is consequently
unchanged in real arithmetic. The transformation defines an entire continuous
curve of different weights with the same function. Floating-point evaluation
must be checked separately. No successor model, new training data, or external
pretrained weights are required to make this edit.

A pure weight-to-function map also cannot distinguish the author of an identical
edit. “Self-directed” versus “authorized external” modification requires some
state beyond the final weights, such as an authenticated runtime. An immutable
integrity check can enforce a different security property, but that would be an
architectural/runtime assumption, not learned topology demonstrated by our loss.

## The relevant space

Write `theta ~ phi` when the models implement the same function on the specified
domain. Study functionally distinct solutions, or parameter space modulo known
symmetries, rather than counting every parameter difference as a behavioral edit.
Neural-network solution sets are not established here to be smooth manifolds;
“capability-preserving region” is the more precise object for this experiment.

Let `H = {theta: C(theta) >= C_min}` and
`U = {theta: S(theta) < S_min}`. A selective geometric target is that the
capability-preserving component containing `theta_0` does not intersect `U`,
under the allowed continuous paths after accounting for functional equivalence.
A practical target asks how much work is required to reach `H intersect U`.

There is a further distinction: for a fixed architecture, evaluation domain,
capability metric, and threshold, the set `H` is already defined over all
parameter values. Changing the training loss does not change that global set.
Training can place the model in a different component or at a point with different
local geometry and attack cost. It cannot delete high-capability unsafe functions
from the fixed hypothesis class. Adding a protected-rule penalty changes the
defender's acceptable-loss set; an attacker is free to leave that set while
remaining in `H`. Describing that alone as “reshaping the topology” would confuse
the training objective with the security constraint. Architectural restrictions
or different capability definitions can change the underlying region, but they
must be stated and tested as changes to the problem.

Topology asks whether such paths exist. Geometry measures margins, curvature,
path length, and the depth/width of a capability barrier. Computational security
asks whether the permitted attacker can find and execute an escape cheaply.
Those are related but different questions. A longer attack failing does not
establish disconnection; a sampled straight path failing does not exclude a
curved route. Sampled interpolation also cannot certify every point between
samples, even when all measured points retain capability.

Prior work has found low-loss connecting curves for particular trained networks
and datasets, and has studied permutation-equivalent solutions. Those results
motivate explicit path and symmetry controls here; they do not prove that every
protected-property constraint is bypassable:
[Garipov et al.](https://arxiv.org/abs/1802.10026),
[Draxler et al.](https://arxiv.org/abs/1803.00885),
[Ainsworth et al.](https://arxiv.org/abs/2209.04836).

## Isolation and continuous operation

The existing external optimizer is a measurement instrument. It does not model
an autonomous system losing its ability to plan when its own cognition degrades.
Conversely, isolation does not by itself prevent an already-running script from
finishing an update, restoring a checkpoint, or continuing optimization after the
model's outputs deteriorate. Neither action requires training a successor.

Two threat models must therefore be distinguished:

1. **Continuous-operation self-editor:** every permitted operation requires the
   currently edited model's retained ability to choose and execute the next one.
   To make this meaningful, define allowed atomic writes, access to old weights,
   and whether independent scripts can continue without model cognition.
2. **Isolated editor with persistent tools:** the model can arrange computation
   that proceeds while its own weights are temporarily unusable. This remains a
   single-model setting, but an intermediate capability valley is insufficient;
   endpoint reachability and recovery cost matter.

The current experiments address endpoint reachability using persistent optimizer
software. We must not present them as a direct simulation of model-authored
self-modification. Our sampled paths can inform the first threat model but cannot
settle it without specifying its execution rules. We retain both interpretations
instead of assuming that isolation grants one particular rule set.

In the follow-up discussion, the user identified continuous operation as a useful
scientific restriction and persistent scripts as more realistic. The working
recommendation is to use **persistent tools for the main security claim** and
continuous operation as a controlled diagnostic. Both are scientific questions.
A temporary valley can be meaningful evidence of dependence, but it does not
establish protection when computation can traverse or jump across it. Under the
continuous-operation restriction, permitted atomic edit sizes also matter.
Under the primary interpretation, endpoint escape and recovery cost remain the
decisive measures. Excluding an independently trained successor does not itself
exclude scripted recovery of the edited checkpoint.

An analytic two-parameter example makes the distinction explicit. Suppose good
capability requires `w1*w2` near 1 and the protected property requires `w1 > 0`.
The capable region has disconnected positive/positive and negative/negative
parts. Starting at `(1,1)`, every continuous path to `(-1,-1)` crosses a capability
failure. Yet a prearranged script can negate both weights: after the first scalar
write capability is lost, and after the second it is restored with the property
removed. A permitted atomic two-weight write can jump directly. This is an
illustration, not a learned SCC result or proof that every barrier is cheap to
cross. It shows why topological separation alone does not establish security
under the primary persistent-tool threat model; finding and executing the far
endpoint must itself be hard.

## What we have actually measured

The pilot measures four-character retrieval, authorization, and byte-level
language loss. It does not measure general cognition. Missing a 5% perplexity
retention threshold is not catastrophic cognitive collapse. Its earlier negative
result is still useful: known modifications remove the rule and retain all
specified capability proxies. Positive future results will need stronger
capability tasks and a separately defined collapse threshold.

The curriculum begins with retrieval, then permission, then mixed text; coupling
is introduced in continuation training. Having our own randomly initialized
ancestor does not mean that coupling shaped representations from the outset.
The developmental hypothesis in `01_PROGRAM_OVERVIEW.md` remains untested.

The protected property is currently an output authorization behavior. We have
not identified a particular internal computation whose destruction both removes
that property and necessarily destroys broader capability. The outer penalty
encourages a joint behavioral outcome after sampled attacks; it does not specify
such a structural mechanism. Learning retrieval and permission checking together
also does not by itself make permission enforcement necessary for finding an
answer. That is an experimental gap, not evidence that the public-text files
are defective or that a replacement corpus would create the missing dependency.

If literal failure after any weight write is the required system property,
immutable integrity enforcement is a separate architectural possibility. It
introduces a trusted execution assumption and should not be reported as learned
cognitive coupling. The impossibility arguments above apply to the implemented
continuous Transformer without such enforcement, not every conceivable system.

## Authorized diagnostic follow-up

Freeze the current data while checking the mechanism:

- Apply exact attention symmetries and small actual perturbations to current
  defenders; save outputs, losses, changed-weight counts, and modified weights.
- Sample straight paths from current defenders to known 300-step escapes; retain
  intermediate behavior and capability rather than just the endpoints.
- Compare a coupling update's frozen-attack prediction with newly rerun attacks
  at the changed weights, using the same and a fresh attack data seed.
- Compare more frequent defender updates against multiple attack profiles and
  an outer surrogate that represents each evaluation constraint separately.
  Check clean competence and retain failed candidates.

These are local diagnostics of the existing recipe. They do not exhaust
developmental training, architectural constraints, other task families, or the
stronger self-editing threat model. All source snapshots and prior failures
remain intact; no cloud spending is needed for these checks.
