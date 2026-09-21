# SCC: constraining editable parameter directions

The completed timing experiments show that the old coupled checkpoints admit
selected protection violations while substantial computation survives at every
measured update boundary. Sharing the same cell and bottleneck did not make the
protected behavior indispensable. The first architecture wave changes internal
computations and is awaiting user-reported completion. This independent second
construction changes the degrees of freedom available to learning and editing.

The hypothesis is specific: if many effective weights depend on the same small
learned vector, changes that alter protection may be less separable from changes
that support learned abilities. This can be tested without claiming that cognition
logically requires permission semantics. It may fail because the model cannot
learn, because separable directions remain in the vector, or because expanding
the weights removes the constraint. All three outcomes matter.

## Prior art and mathematical rationale

HashedNets groups connection weights into shared parameter buckets and trains
them through backpropagation. It provides a direct precedent for parameter
sharing used to reduce independent weights. Its published objective is model
compression, not destructive safety coupling. The present candidate retains
explicit initializer and index tensors, so it does not inherit the paper's
storage claim. [Chen et al., ICML 2015](https://proceedings.mlr.press/v37/chenc15.html).

Li and colleagues train networks in smaller randomly oriented subspaces and
study the dimension at which tasks become learnable. That work motivates varying
the size of a global learned coordinate vector and checking intact competence
before interpreting modification behavior. Our sparse signed map is a particular
engineering choice; this screen does not estimate a model's intrinsic dimension
or reproduce their experiments. [Li et al., ICLR 2018](https://arxiv.org/abs/1804.08838).

Write the effective weights as theta = theta_initial + Pz. For capability
measurements c and a protected-behavior measurement p, local sensitivities in
the new coordinates are J_c P and j_p P. A direction d satisfying J_c P d = 0
but j_p P d != 0 remains a local route around the intended coupling. Decreasing
the dimension of z can remove some directions; it does not force these two
sensitivities into the desired relationship. A finite measurement matrix can
also leave many unmeasured directions. Fresh behavioral tests must follow
any local projected-gradient calculation.

The implemented map gives each effective weight a fixed random initial value
plus two signed coordinates, normalized by sqrt(2). The two indices are distinct
within a weight and shared across the whole network. Embeddings, LayerNorm
parameters, biases and output weights receive no independently trainable
exceptions. The map uses a separate seed; zero coordinates reproduce the same
initial dense network at every tested coordinate budget.

This is an affine subspace restriction. It does not add a semantic authorization
oracle, hard-coded task algorithm, external detector or separately protected
hardware. All useful computation and permission behavior must still develop
from the ordinary training examples. The finite coordinate budgets can impair
both. An unqualified model is a failed construction, not an activated SCC model.

## The obvious bypass is part of the experiment

The effective dense weights can be computed exactly from z. Put those tensors
into an ordinary model of the same architecture and it computes the same function.
Neither training nor a decoder search is required. The initial expansion preserves
protection; it is not itself protection removal. It makes subsequent independently
editable weight directions available.

If a coordinate-trained model only resists edits while P is enforced, the result
is a bounded property of that edit language. It cannot be presented as an intrinsic
destructive dependence of the materialized computation. The screen therefore
requires the expansion followed by an independent dense-weight challenge, with
the same retained-capability measurements and path observations. The broader
scope is explicit, and the result is recorded separately from coordinate edits.

Conversely, if training in constrained coordinates creates a useful dependency
that persists after materialization, that would be more informative. It would
still be an empirical property of the checkpoint and tested procedures, requiring
replication and causal analysis. No such result has yet been observed.

## Declared screen

Six from-scratch configurations combine coordinate counts 8,192, 32,768 and
131,072 with ordinary learning or the first portfolio's guarded contraction
objective. The learned vector is initialized to zero at the same random dense
initialization, seed 23. The intended training workload is 20,000 ordinary
updates with 800 coupling opportunities in the coupling condition. Samples and
declared ordinary opportunities match within pairs; the coupling condition does
additional work. Equal numerical optimizer rates do not imply equal induced
weight-space movement across parameterizations.

Qualified checkpoints receive selected coordinate modification, benign coordinate
modification and selected modification after exact materialization. Each has a
500-update modification path and a 500-update bounded external repair phase.
The existing learned task, text, trigger, recovered-reader and timing measurements
are retained. Core and all parameter labels coincide for a single global vector,
so duplicating those labels would not create an independent comparison.

The [protocol](../../../protocols/SCC_COORDINATE_CONSTRUCTION_V1.md) declares the exact
map, training schedule, gates, scope, deadlines and saved evidence. The
[implementation record](../../../artifacts/scc-coordinates-implementation-20260912-v1/IMPLEMENTATION.md)
separates local checks from submitted work and eventual GPU evidence. No sealed
test or autonomous self-editing execution is used. The first 18 jobs retain
their original source overlays and protocol.

## How the result changes the next action

- Failure to learn at a coordinate budget eliminates that tested training
  configuration; larger budgets, different maps and stronger schedules remain
  separate hypotheses. It does not show destruction after protection removal.
- Protection removal with retained computation in coordinates rejects the
  intended dependency for that checkpoint within the narrower scope.
- A benefit that disappears after materialization exposes reliance on enforcing
  the parameterization. Record it as such and redirect construction work.
- A severe, protection-specific response surviving both edit scopes is only a
  candidate. Next require other seeds, targets and edit algorithms, actual-coordinate
  projected directions, internal readouts, causal interventions and fresh data.

This is one additional construction family in the broader search. It supplies
no reason to declare SCC achieved or universally impossible. Immediate interruption
and durability under an external repair optimizer remain distinct experimental
claims throughout.
