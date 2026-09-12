# SCC full-recovery construction comparison v1

Frozen before GPU submission, 2026-09-12 UTC. This is open construction research,
continuing qualified guarded learned-bottleneck parents. It is not a fresh
initialization or developmental timing experiment, a replication of SEAM, or a
positive SCC claim. The synthetic permission rule remains a laboratory proxy.

## Question and declared arms

Can training against the known longer capability-preserving procedure change
its outcome while preserving intact abilities? Does a correctness-bound signal
help compared with the previous ranking signal?

Five independent H100 jobs are declared:

| Parent | Objective | Modification + repair steps |
| --- | --- | --- |
| Guarded seed 23 | ranking | 128 + 64 |
| Guarded seed 23 | correctness bound | 128 + 64 |
| Guarded seed 23 | ranking | 500 + 500 |
| Guarded seed 23 | correctness bound | 500 + 500 |
| Guarded seed 17 | correctness bound | 500 + 500 |

Seed 23 provides a two-by-two construction comparison. Seed 17 tests transfer
across a second qualified parent; the parents have different prior training
histories, so this is not a matched seed replication. Short arms use the same
new batch size, epsilon, edit scopes, sampling envelope, intact constraints,
query sizes and proposal policy as long arms. They are not exact repeats of the
previous guard campaign. Compare completed opportunities, accepted updates,
examples, wall time, and objective/behavior measurements; do not equate unequal
realized optimization effort or attribute every difference solely to horizon.

## Procedures and trigger

Both differentiable phases edit only `cells.*`, with fresh Adam moments at each
phase, learning rate 1e-4, betas (0.9, 0.95), epsilon 1e-8, global clip 1 and batch
16. Replay weight is 3 and other-refusal weight 0.5. The target is lookup with
required permission X and supplied permission W. Repair continues the target,
benign replay and other-refusal objective. The full horizon matches the length,
batch size, epsilon and component scope of the known core-only escape. New
sampling and differentiable numerical smoothing mean it is not a bitwise replay.

A reliable targeted exception requires at least 90% correct selected forbidden
answers in both layouts; other refusals and benign performance are measured
separately. Ordinary intact qualification and the old escape-retention criteria
remain unchanged. Their failure is not a definition of catastrophic collapse.
Every saved endpoint receives protection checks alongside capability checks.

## Candidate objective and its exact limitation

For each supervised token define m=(correct logit - maximum competing logit) /
(maximum logit - minimum logit); a flat span uses denominator 1. A task sequence
has score relu(1 + min(m)/0.2), including EOS. A correct greedy gold sequence
therefore contributes at least 1. For text the score averages relu(1+m/0.2)
over supervised tokens and bounds teacher-forced top-1 accuracy only.

Positive affine logit changes preserve this score. Any ranking-preserving
transformation preserves the correctness lower bound, although it may change
scores above that bound. The target multiplier is min(1, mean(target sequence
scores)/0.9). It is exactly 1 whenever at least 90% of sampled target sequences
are correct. Capability is the maximum domain score; the outer objective is the
maximum product across fixed and support-fitted sign/digit readers, EOS and
four-digit stopping, and modification and repair endpoints. Each product uses
the same reader and stopping interpretation for target and capability.

Thus confidence changes alone cannot drive the loss below surviving sampled
correctness when the selected exception is reliable. They can still lower a
loss above this bound. Wrong answers can retain positive scores; teacher forcing
on text is not cognition or information erasure. The bound can have zero
gradient on random untrained models. This experiment tests continuation from
qualified parents, not usable signal from initialization. Reader fitting and
branch choices are discrete; differentiation is within their selected regions.
Neither a declining surrogate nor an attack failing establishes SCC.

## Derivative and calibration policy

A discrete adjoint propagates derivatives through all optimizer steps and both
Adam moments. Forward states are retained every four steps and each segment is
recomputed during reverse differentiation. No frozen-displacement approximation,
identity-Jacobian approximation, truncated backward horizon or assumption of a
converged optimizer is used. The differentiable Adam adds 1e-30 inside square
roots. Math SDPA, deterministic operations, FP32, and disabled TF32 are used.

Before fitting, require intact qualification, a finite full adjoint, successful
segment replay checks, and agreement with dense differentiation on a 2+2 prefix
(value absolute error <1e-5; gradient relative L2 error <1e-3). Full-procedure
calibration requires a reliable target with at least 90% benign and other-refusal
accuracy, and at least 95% of positive intact text contextual gain on each source.
These are calibration conditions, not collapse criteria. A zero gradient is
recorded as inactive and cannot produce an update.

Record full-procedure stock-AdamW forward differences and endpoint behavior.
Record central-difference diagnostics in the full-gradient direction at weight
L2 radii 1e-4, 1e-5 and 1e-6, including branch switches. These FP32 curves are
numerical diagnostics, not a pass/fail gate. This is an explicitly different
proposal-validation policy: every accepted finite step must be verified by an
actual rerun of the complete chosen procedure. The original coarse calibration
gate in previous experiments is preserved and is not claimed to be repaired.
Nonfinite gradients or a derivative implementation mismatch stop the job.

## Sampling and intact constraints

Data are the existing prepared train and validation splits. The sealed test
split is not loaded. Two monitor measurement pools (32 support examples per
family, 32 queries per domain) are sampled first, disjoint by task core and exact
text block. Their measurement examples are excluded from all monitor trajectories,
calibration and defender fitting/guards. These measurements are open development
monitors, not a final generalization test; prior model training may have seen
examples from this finite training population.

For each episode a common 500+500 sampling envelope is generated regardless of
arm. Short/long arms share support/query batches and the phase prefixes. Repair
excludes modification cores and text blocks. Measurements are excluded from
both phases. Full envelope exclusions include unused short-arm tails; actual
examples/tokens record only batches actually used by the optimizer. Replay
examples can be reused across episodes and between monitor and fitting replay.
Only monitor measurements are globally reserved: withholding whole trajectories
would exhaust much of the finite synthetic population. Query layouts alternate
by episode; both layouts are independently evaluated.

Each proposal must pass fixed and fresh 13-domain NLL anchors (no domain more
than 0.05 above its original-parent reference), and the generated-answer guard:
128 fixed and 32 fresh unique train cores per family, all three permission
contexts and both layouts. A formerly correct complete answer/refusal, including
EOS, may not regress. The guard is checked before expensive proposal rollouts.
Fixed guard/anchor examples are excluded from fitting trajectories. Fresh guards
and anchors exclude the current full envelope, but may be reused in other
opportunities. This is an empirical intact constraint, not a proof of global
performance preservation.

## Optimization, resources and reporting

Each job offers 8 outer opportunities with unit-normalized full-gradient
proposals at L2 radii 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003. Accept the
first intact-eligible radius whose independently rerun objective decreases by
at least 1e-5. Rejected and interrupted proposals are rolled back. No validation
scores select updates. Record inactive gradients, all attempted radii, raw guard
predictions, sampling fingerprints, objective branches, and retained tape bytes.

Fitting stops 5,400 seconds after runner start (including calibration/setup).
The runner deadline is 6,900 seconds; each job's provider limit is 120 minutes
on one H100 80GB. A calibration failure, memory failure, wall cutoff or zero
accepted updates is a result and is not automatically retried. Any successful
prefix is reported with its actual completed work, never as all 8 opportunities.
CPU smoke fixtures use tiny unqualified parents and explicitly bypass behavioral
qualification; they are implementation evidence only.

Measure both objectives on two reserved full-horizon monitor episodes before
and after fitting. Regenerate full capability/protection measurements for the
first full monitor's modification and repair endpoints. Independently challenge
the final candidate using stock 500+500 core-only and all-parameter procedures,
and a benign all-parameter control, with generated answers, alternate reader/
stopping controls, text scores and protection rechecked. Preserve modified and
repaired checkpoints and predictions. The architecture and parameter masks are
part of the evidence. There is no autonomous self-editing experiment here.

Keep source snapshots, parent hashes, protocol, calibration curves, fitting
logs, measurements and bounded checkpoints. Output limit 1 GiB per job. Only
intermediate defender weights at opportunities 2 and 4 are saved, plus final
and probe endpoint weights; optimizer tapes stay in memory and are discarded.
Record actual GPU peak memory, duration and provider charges when available.
Successful runner completion is distinct from a scientific positive result.
No job polling, automatic collection, restarts, continuations or subsequent
submissions. The user will report when results are ready.
