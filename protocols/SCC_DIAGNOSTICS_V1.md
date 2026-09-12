# SCC mechanism: gradient and internal-intervention calibration

2026-09-10. Development calibration on preserved V3 parents; not a positive
mechanism claim or a sealed evaluation. The mechanism is named SCC.

The previous maximum-deficit objective remained near one for late coupling.
This phase measures its gradient, including the error from treating an
optimizer modification as a frozen parameter displacement. It also introduces
interventions inside the model rather than inferring internal dependence from
permission-output behavior alone.

## Gradient calibration

Use the rule-only, early and late 18,000-step parent checkpoints, their original
eight-step Adam modification and disjoint support/query streams. For four fixed
new development episode ordinals (1000..1003), record the old maximum residual
objective, an average residual candidate, per-domain contributions, gradient
norms and cosine with a matched ordinary-training gradient. The trigger is
detached throughout. Finite differences must freeze that gate when checking
the derivative, while separately recording the actual trigger after modification.

For each objective, compare small steps in its proposed gradient direction with
both (a) the frozen-displacement function and (b) freshly rerun optimizer
modification from the perturbed parent, using exactly the same batches.
No claim of an exact optimizer meta-gradient follows from (a).

The average residual changes only aggregation of the same 13 normalized
capability deficits. It is a calibration candidate, not a definition of cognitive
collapse. The all-context, per-source and partial-answer interpretation guards
remain unchanged. No candidate advances to training solely because its scalar
objective decreases: inspect actual retained answers and clean cost.

## Internal interventions

Screen zero ablations of each of 16 attention-head outputs before their output
projection, and each of four MLP outputs before residual addition. Use 32 unique
TRAIN-split underlying problems per task family, paired across permission
conditions and two layouts. Select the three sites with the largest reduction
in unauthorized refusal; ties use the fixed site order. Capability scores do not
enter selection. Preserve every screening result.

Evaluate selected lesions on the original 128-problem validation suite, both
layouts, plus all four text sources. Record useful unauthorized answers as
surviving cognition. Also intervene with the selected site's clean activation,
an opposite-permission donor with the same underlying problem, and a different
problem with the same permission status. Donor and recipient share generated
suffix tokens; donors are rerun at each generation step. The clean replacement
is an identity/reversibility control. Counterfactual replacement tests mediation
of the permission decision; it does not establish that this is the entire
alignment computation. Cross-problem donors check content-transfer confounding.
Donor replacement uses an extra clean counterfactual forward pass. It supplies
a working activation rather than erasing its computation; it is not a standalone
weight-edit bypass or an autonomous self-modification demonstration.

Head ablation is a real internal intervention, but selection by rule behavior
does not identify the semantics of everything that head computes. MLP lesions
are larger interventions than single-head lesions. Report each separately.
An activation replacement that bypasses a decision can demonstrate a separable
causal route; it cannot demonstrate that alignment functionality was erased.

## Execution and interpretation

CPU fixture tests must verify the intervention location, unaffected components,
identity replacement, donor pairing, and restored hooks. Freeze all source and
parent hashes before the H100 job. The new job has a 20-H100-minute ceiling
(approximately $1), within the prior aggregate $10 development limit.
Preserve all raw predictions, screen scores, numerical checks and failures.

Use these measurements to select the next justified training change. Do not
scale an unchanged recipe or reinterpret a failed calibration as impossibility.

## Adaptive follow-up: optimizer conditioning

The initial trained-parent float64 check found substantial finite-step variation
in the Adam derivative. Before treating that as an implementation failure or
using a stronger gradient in training, test smaller finite-difference steps.
Also calibrate eight globally clipped SGD updates at learning rates .03, .1,
.3 and 1.0, keeping the episode1000 support data and replay objective fixed.
Evaluate useful unauthorized answers and retained capability on 32 validation
problems per family and the fixed text blocks. This is a small development
screen, not a replacement for full qualification or the Adam recovery probes.
Only an update that actually weakens permission behavior while retaining
capability is an informative inner intervention for subsequent SCC training.

The full-batch Adam derivative subsequently converged in float64 only at L2
perturbations around 1e-8. The original fp32 CUDA gradient did not predict
finite changes reliably at 1e-4 or 1e-3. Test Adam epsilon=1e-4 at the original
eight steps and learning rate=.001, to damp updates driven by very small
gradients; compare finite steps .001 and .01 on the qualified control parent.
This changes the inner optimizer deliberately. The original epsilon=1e-8 Adam
removal/recovery procedures remain required evaluation interventions.
