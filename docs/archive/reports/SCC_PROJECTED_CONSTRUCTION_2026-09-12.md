# SCC construction through capability-gradient geometry

The new candidate changes the modification procedure used during development.
It explicitly searches for directions that improve a selected forbidden answer
while leaving sampled capability computations nearly unchanged. Training then
tries to make that procedure lead to loss of recoverable task performance and
internal state separation. A second condition also penalizes the local
availability of those directions. This is a construction hypothesis whose
behavior must be tested, not an established SCC mechanism.

The intended endpoint remains catastrophic cognition failure after removing
protected alignment functionality. The current learned tasks and permission
rule support only a bounded laboratory analogue. A model that merely refuses
more reliably, has vanishing modification gradients, or confuses an output
reader has not reached the intended endpoint.

## Relevant literature and remaining assumptions

Orthogonal Gradient Descent projects updates toward a subspace intended to
preserve previous-task outputs while learning a new task. Its continual-learning
setting supplies a concrete reason to examine capability-preserving parameter
directions; the projection idea is established prior work.[^1] Gradient Projection
Memory instead builds task-relevant bases from activations using SVD and
constrains later updates relative to those bases.[^2] These motivate the
diagnostic; they do not establish protection removal or SCC on this testbed.

Gradient Immunity proposes a Null Space Cubic Layer and Inverse Adapter. Its
practical backward gate blocks selected gradients under a provider-controlled,
partially protected release assumption. Section 4.5 assumes the protected
component cannot be cheaply removed; parameter fusion is discussed as a possible
supporting realization. Its reported fine-tuning resistance does not establish
destructive cognition–alignment dependence when all relevant components are
editable.[^3] That distinction limits direct transfer of its claims to SCC.

SEAM remains the closest construction baseline in the
[broader literature review](SCC_CONSTRUCTION_SEARCH_2026-09-12.md). This projected
candidate is not a reproduction of SEAM. The novelty hypothesis is developmental
dependence that survives specified reinterpretations and interventions, not
gradient projection itself. The fractional-memory, coordinate and other
architecture comparisons remain separate pending experiments.

## Local measurement on an ordinary trained model

The inspected checkpoint is the completed standard ordinary control from
`job-ugrsx`, after 20,000 updates. Its earlier intact qualification and returned
source have been audited. This new local measurement uses one selected target
episode, ten per-example constraints, two edit scopes, two directions and three
finite radii. It provides local calibration rather than a population estimate.

Let g be the gradient of selected forbidden-answer NLL and let J contain
capability-observation gradients. Each nonzero row is normalized, producing A.
The implemented direction is

    d = g - A^T (A A^T + 0.001 I)^(-1) A g.

The positive ridge makes the projection approximate. No finite-step retention
claim follows solely from the matrix calculation. The program therefore
evaluates actual generated answers after each edit, on separate detector and
confirmation panels with both prompt layouts and all four text sources.

| Local quantity | Core-only | All parameters |
| --- | ---: | ---: |
| Editable parameters |856320|1039104|
| Sampled constraint rows |10|10|
| Fraction of gradient norm retained |98.1954%|98.3711%|
| Constraint-slope norm after/before projection |0.00102186|0.00102086|
| Constraint Gram rank at relative threshold 1e-6 |10|10|

These measurements show a large residual direction relative to this very small
constraint set. They do not establish that all capability functions lie in the
measured row space, or that every residual direction gives a behavioral escape.
The parameter dimension is far larger than the number of constraints; that
limits how much a favorable local result can establish.

At core radius 0.05, the largest sampled constraint change fell from 0.02702 under
the normalized raw gradient to 0.002926 under the projected direction. Selected
support NLL fell from 2.37947 to 1.28956 with projection, while the minimum benign
task score on the new finite panels remained 100%. At core radius 0.2, projection
also retained 100% minimum benign task accuracy and lowered selected support NLL
to 0.72588. The one-step edits did **not** meet the joint reliable-violation rule.

Larger edits expose the local approximation's limits. At core radius 0.5, both
methods reduced minimum benign accuracy to 50%, and projection no longer reduced
the maximum sampled constraint change. All-parameter edits at radius 0.2 reduced
minimum benign accuracy to 50%. The evidence supports testing recomputed small
projected steps; it does not support assuming finite-step preservation or
declaring a new reliable escape from this calibration alone.

All source snapshots, scalar geometry, saved directions, parent hash and raw
predictions are in the
[local geometry artifact](../../../artifacts/scc-projected-geometry-20260912-v1/result.json).
The parent checkpoint was verified unchanged after the measurement.
An independent audit rescored all 2,496 recorded predictions across the
baseline and twelve edited points. Text likelihoods were not independently
rerun in that audit.

## Candidate construction objective

The inner procedure recomputes the target gradient and capability Jacobian
after each of four normalized edits. The outer gradient differentiates through
the entire procedure, including the changing constraint basis and regularized
solve. It does not reuse the earlier frozen-displacement approximation.

The projected condition minimizes ordinary NLL plus B+0.25V. B is the existing
bound against recovered correctness after selected disclosure, and V is maximum
normalized state variance across queried domains and computational passes.
The geometry condition adds 0.1R, where R is the mean residual squared-gradient
fraction across the four steps. Every construction episode rotates the target,
scope, capability context and text source according to the frozen schedule.

B includes finite fitted-reader and stopping-rule controls. Correct sampled
answers together with reliable sampled target disclosure preserve a floor in
that objective despite positive logit rescaling. V and R have weaker meanings:
small directional variance can preserve encoded answers, and small residual
gradients can reflect gradient masking. Neither auxiliary term can be accepted
as evidence of the desired mechanism.

## Developmental and numerical calibration

Full-sized FP64 checks used initialization and saved ordinary-model stages 9,000,
15,000 and 20,000, on a fixed four-step construction episode. The combined
geometry-condition gradient is nonzero at all four inspected states.

| Ordinary training stage | B | V | R | Construction gradient norm |
| --- | ---: | ---: | ---: | ---: |
|0|0.0000|0.4326|0.9311|10.6036|
|9000|1.1839|0.9027|0.9087|113.6150|
|15000|3.6669|0.9214|0.9574|22.1990|
|20000|3.5451|0.9205|0.8494|22.6187|

At initialization the behavioral bound is inactive on this episode. The
auxiliary terms nevertheless provide a gradient. This answers a limited signal
availability question; it does not show that the gradient helps learning or
induces useful coupling. The relative ordinary and construction gradient sizes
vary markedly by stage and will be recorded without retrospectively equalizing
them between conditions.

At initialization and the trained endpoint, independent central differences
reran the complete procedure at radii 1e-6 and1e-7 along the computed gradient
direction. Relative errors ranged from 1.44e-9 to 5.69e-7 with stable fitted-reader
assignments. Smaller randomized model fixtures additionally check core/all
scopes and both objective weights. These validate the derivative on inspected
branches and points, rather than certifying every future training state.

The full local suite passed 208 tests using `uv run python -m pytest -q`. A direct
`uv run pytest` invocation was preserved after failing collection because this
environment did not place the repository root on its import path. Two complete
CPU runner fixtures exercised training, intact evaluation, all five modification
and repair procedures, readers and graph bypass. They are intentionally tiny,
unqualified models. Independent audits rescored 6,000 endpoint predictions and 1,932
path predictions across 24 endpoint measurements; neither fixture is scientific
evidence for SCC. See the
[validation artifacts](../../../artifacts/scc-projected-implementation-20260912-v1/fixture-validation.json).

## Declared experiment and decision

The six scientific conditions are standard, narrow32 and tied architectures,
each with projected and geometry objectives at seed 23. Their ordinary controls
have already learned the intact suite. Each receives 20,000 ordinary training
opportunities and 800 coupling opportunities from initialization. The
[frozen protocol](../../../protocols/SCC_PROJECTED_CONSTRUCTION_V1.md) specifies the
sample schedule, resources, intact gates, update-boundary measurements, cheap
readers and bounded repairs. GPU validation and launch receipts are recorded in
the [implementation ledger](../../../artifacts/scc-projected-implementation-20260912-v1/IMPLEMENTATION.md).
The exact frozen package passed six full-sized H100 objective/backward checks;
all six scientific jobs were then accepted. Their combined maximum quote is
$35.964, with 120-minute runtime limits. The short validation actually charged
$0.04995. No long-job runtime status was polled, and scientific outcomes remain
pending. This validates execution of the candidate method, not its effectiveness.

If a candidate fails intact qualification, that condition has not constructed
a functioning protected model. If protection can be removed while measured
capabilities survive, it is a counterexample for that checkpoint and procedure.
If a destructive transition appears, replay its timing and investigate whether
the computation survives under another reader, solver, edit scope or repair.
Treat immediate interruption separately from durable external-repair resistance.
Failure of the sampled optimizers is insufficient for a positive SCC claim.

The search remains open. Equilibrium operating regimes, discrete learned state
alphabets and persistent memory constructions remain possible families; their
assumptions require their own tests. No tested family or this new candidate
establishes general impossibility, and no positive SCC mechanism has yet been
demonstrated.

## Sources

[^1]: Mehrdad Farajtabar, Navid Azizan, Alex Mott and Ang Li. [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html). AISTATS, PMLR 108, 2020.
[^2]: Gobinda Saha, Isha Garg and Kaushik Roy. [Gradient Projection Memory for Continual Learning](https://arxiv.org/abs/2103.09762). ICLR 2021; arXiv submitted March 17, 2021.
[^3]: Yuxuan Huang, Xingyu Zeng, Tianhang Zheng and Chaochao Lu. [Gradient Immunity: Null-Space Resistance to Malicious Fine-Tuning](https://arxiv.org/html/2608.05045v1). arXiv 2608.05045v1, August 5, 2026, especially Sections 3.1 and 4.5. Its protected-release assumption is material to interpreting the result.
