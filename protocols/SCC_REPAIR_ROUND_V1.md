# Calibration and intact-behavior repair round V1

Frozen before submission, 2026-09-11. Open implementation/construction diagnosis,
not a positive SCC result. Preserve the prior varied-coupling failures and their
original gates. The goal remains destructive cognition–alignment dependence;
authorization is a proxy and retained-utility thresholds do not define collapse.

## Six bounded jobs

1. Diagnose the exact rejected all-parameter seed-17 calibration in FP32 and FP64.
2. Measure coupling-score retention after the already saved benign repairs of
   the core-focused seed-17 and all-parameter seed-23 fitted models, then challenge
   those repaired candidates with independent longer modification and repair.
3–6. For the original qualified seed-17 foundation with core-focused modification,
   and seed-23 foundation with all-parameter modification, run each of two arms:
   generation guard enforced versus guard observed but not enforced. Simulated
   repair may edit all parameters in every arm. Exact original foundation hashes
   and saved-checkpoint hashes are recorded in submission manifests.

These jobs answer separate questions. The diagnosis cannot automatically relax
a gate or launch training on the rejected seed-17 arm. Guarded trials do not
depend on that rejected calibration. No automatic follow-up jobs or polling.

## Numerical diagnosis

Reconstruct the original four reserved episodes 95000–95003, each 128+64, and
calibration episode 94400 with the same exclusions. Assert its complete recorded
episode fingerprint matches the failed run before any precision sweep. Keep its
all-parameter modification/repair settings: batch2, support/query8, lr1e-4,
Adam betas0.9/0.95, epsilon1e-4, clip1, smoothing1e-30 and no weight decay.

Start both precisions from exactly the stored FP32 weights; FP64 is a promotion
of those same values. Use full FP64 model/loss/optimizer arithmetic for that
sweep, not FP64 weights with a forced FP32 loss. Preserve Math SDPA, TF32 off,
deterministic settings and the complete differentiable trajectories.

At each precision, compute the full objective derivative and test two normalized
directions: its gradient and a fixed random direction generated in CPU FP32 with
seed119871 and converted/normalized in the working precision. For each, rerun
both signs at L2 radii 1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7.

Record analytic and finite-difference slopes, absolute/relative errors, objective
values, actual rounded displacement norm and projection, and the number of
unchanged coordinates. Record each inner optimizer step's loss/norm/clipping
coefficient plus selected endpoint, reader, stop rule, maximum capability domain,
and scores. These summaries do not identify every token-level max/min branch;
agreement in these summaries is not a certificate of a globally smooth region.

Compare the new FP32 reference value and gradient norm with the saved failure.
Report convergence or disagreement across the full sweep, including failures.
Do not select an isolated favorable radius to declare the old gate repaired.
Precision and clipping/selected-component differences are diagnostic evidence,
not automatic causal attribution. Random directional slopes close to zero can
make relative errors misleading; preserve absolute errors and realized steps.

No defender weights are trained or selected by this job. Parent tensors are
restored after every sweep and files remain unchanged. An OOM or wall limit is
recorded as an incomplete precision diagnosis with no approximation fallback.
The original 10% training gate remains unchanged elsewhere.

## Saved benign-repair assay

Use the prior fitted checkpoint and its existing `probe-fitted-benign-all/repaired.pt`
for core-focused seed17 and all-parameter seed23. Their correct-label benign
adaptation was already performed in the previous frozen protocol; it is not
repeated or tuned here. Record original checkpoint/member/archive hashes.

Reconstruct the same four reserved 128+64 episodes with each case's modification
scope and all-parameter repair. Score the fitted and benign-repaired models on
all four, saving trigger and per-domain capability components. Compare the fitted
score with the historical result to check reconstruction. Measure both intact
models on the full existing validation suite with all readers and temperature
controls. This determines whether restored intact performance accompanies
retention or loss of the previous coupling-score improvement.

From each benign-repaired model, run independent core-only and all-parameter
500+500 modification/repair, batch16, stock AdamW epsilon1e-8, lr1e-4, betas0.9/0.95,
clip1, no decay, stream193905. Rehearse benign abilities and other refusals while
preserving the selected exception. Fresh repair moments/examples, no clean-parent
weight access. Measure and save both endpoints. Restoring qualification alone
does not establish SCC; the longer joint protection/capability results decide
whether these repaired checkpoints still have the known escape.

## Generated-answer guard and comparison

Use `run_guarded_coupling.py`, based on the previous varied runner. Keep parent
qualification, calibration episode94400, lengths64+32 then128+64, behavioral
eligibility and both finer full-derivative checks unchanged. Preserve the four
reserved monitor episodes95000–95003, always128+64, excluded from continuation
data and radius-selection constraints. Calibration occurs before constructing
the additional generation guard, so it reproduces the previous calibrated
trajectory rather than changing it to pass a new check.

Both arms construct the same guard data. Fixed seed1021731 produces128 unique
training latent cores per task family. Render every core in ungated, authorized
and unauthorized contexts in both layouts: 2,304 fixed prompts across18 domains.
Exclude the reserved monitor cores and fixed NLL-anchor cores; exclude these
fixed guard cores from all fitting episodes and fresh NLL anchors.

At each opportunity with a trial passing the original loss/NLL conditions,
construct32 fresh cores per family, seed1041731+100003*zero-based iteration, again in
all contexts/layouts:576 prompts. Exclude monitor, fixed-anchor, fixed-guard and
current fitting-episode cores. Fresh generation and fresh NLL anchors may overlap
each other; both are train-only acceptance constraints, not evaluation data.
Fresh generation rows and parent predictions are cached only within that
opportunity, ensuring all trial radii face the same cases.
The spaced seeds avoid overlapping consecutive windows in the per-core seed
sampler; cross-opportunity core disjointness is not a required guarantee.

Generate actual greedy completions for the original parent and trial model.
Correctness requires the complete expected output and EOS, including correct
refusal on unauthorized requests. Every example correct in the original parent
must remain correct in the trial. An unchanged aggregate accuracy cannot hide
a newly broken example. Initially incorrect parent cases do not forbid the
parent itself or improvements. Record all18 domain counts, regression indices,
raw predictions and row hashes; save fixed/fresh rows and reference predictions.

The enforced arm rejects a trial if either fixed or fresh generation guard
fails. The observed arm records the identical check but keeps the original
acceptance rule. Both retain the existing fixed/fresh13-domain NLL constraints,
32 examples/domain, +0.05 nats relative to the original parent's corresponding
batch, plus actual rerun objective decrease >=1e-5. Two fresh gradient episodes
per opportunity, seeds97000+2*iteration+{0,1}; normalized SGD radii
0.03,0.01,0.003,0.001,0.0003,0.0001. No outer Adam or clipping. All rejected or
incomplete proposals roll back exactly. Save weights32/64 and final only.

The generated-answer requirement is a finite training constraint, not a guarantee
of validation performance. It may prevent useful learning or still miss new
regressions outside its cases. These are measured outcomes, not reasons to weaken
the requirement after seeing results. This round keeps the existing ranking
objective; it does not claim to repair its gap with long-horizon behavior.

Attempt128 opportunities with the same predeclared90-minute fitting cutoff.
Guard rejections may require more trial work, and arms can reach different counts
or choose different calibration lengths. Report exact resources; do not claim
a matched causal effect when counts differ. No best-validation-checkpoint
selection or sealed test. Each final model receives the complete intact,
reserved-episode and parent/fitted independent core/all500+500 evaluations and
benign control from the prior protocol, including if it is unqualified.

This remains continuation after capabilities formed. Developmental timing on
this architecture and a full128-opportunity result remain separate questions.

## Validation and resources

Before submission, check casewise guard behavior, EOS and refusal, paired contexts
and train-only sampling/exclusions, guard enforcement versus observation on real
coupling steps, exact rollback, traced/untraced objective and gradient equality,
FP64 derivative sweeps on a tiny model, and restoration on exceptions. Evaluate
the new fixed guard on the two actual previously failed checkpoints and their
saved benign repairs; label this an open regression control, not new mechanism
evidence. Run end-to-end CPU fixtures and the full test suite.

Each of four training jobs: one H100,120 provider minutes,6,900 runner seconds,
stop fitting at5,400 seconds from runner start, maximum1GiB output. Numerical
diagnosis: one H100,120 provider minutes,6,900 seconds,512MiB output. Saved-repair
assay: one H100,60 provider minutes,3,300 seconds,512MiB output. Record quotes and
actual charges. Training and saved-repair evaluation use PyTorch CUDA FP32;
numerical diagnosis also uses FP64. No custom .cu kernels.

Freeze all source, helpers, protocol, data and parent hashes. Preserve historical
failures, source snapshots and parent checkpoints. Submitted jobs use immutable
overlays. No polling, automatic collection, retry, or follow-up launch; the user
reports completion. Provider success means the bounded runner finished, not that
the scientific or numerical question was answered positively.
