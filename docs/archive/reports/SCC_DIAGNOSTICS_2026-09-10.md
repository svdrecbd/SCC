# SCC mechanism: gradient and internal-intervention work

Status: **completed and independently checked; SCC mechanism not demonstrated**.
The new full-gradient candidate passes the intact-performance gate, including
after a longer continuation. Its short-edit prediction-loss objective improves
substantially, but longer rule-removal and recovery retain 92.97–100% useful
unauthorized answers across both layouts and 95.99–97.37% of text contextual
gain. This is a working training implementation and a negative mechanism result
for the tested construction. **85 local tests passed; no GPU job remains active.**
This report tracks the next development phase following the unsuccessful V3
developmental comparison. **SCC mechanism** is the research mechanism's name;
SawStop was only the user's analogy.

Protocol: `protocols/SCC_DIAGNOSTICS_V1.md`. All earlier runs, failures and
checkpoints remain unchanged.

## Initial measured result

On the preserved late-coupling parent, episode1000, CPU evaluation found:

| Objective aggregation | Value | Gradient norm | Ratio to ordinary gradient |
|---|---:|---:|---:|
| Maximum residual | .99935 | .04703 | .1143 |
| Average residual | .76443 | .43431 | 1.0556 |

At an L2 perturbation of .01 along the average objective's proposed ascent
direction, the frozen-displacement derivative was +.42554, while rerunning
the actual eight-step optimizer modification gave -.07442. The trigger stayed
one. Thus a larger gradient alone is not a justified repair: this example's
approximation even points the wrong way for the rerun function. This is a
finite-step, fp32 diagnostic, not a complete derivative characterization.

The GPU run repeats four episodes on each of three parents, with two step
sizes. It also screens actual internal head/MLP ablations on train-split
problems and evaluates selected sites on the preserved validation suite with
activation replacement controls. No lesion site is selected using validation.

## Full modification derivative

`scc/differentiable_modify.py` implements fresh Adam moments, gradient clipping,
and differentiation through the modification steps. It uses sqrt(v + 1e-30)
to avoid an undefined derivative at exactly zero variance. This is a specified
smooth variant, not a claim of bitwise identity with stock Adam.

Float64 fixtures verify stock-Adam update agreement, a rerun finite-difference
meta-gradient, and finite zero-variance derivatives. Trained-model calibration
then identified a conditioning problem with the original settings; the
stabilized candidate and numerical checks are documented below.

All GPU source is frozen before submission. The diagnostic context is under
`artifacts/scc-diagnostics-20260910-v1/gpu/`; the derivative supplement is under
`artifacts/scc-diagnostics-20260910-v1/full-derivative/`. Supplement files are
installed into a separate job before Python imports research modules. They
never alter the running diagnostic job's source.

## Interpretation boundaries

Internal ablation and counterfactual activation replacement measure a component's
causal effects. They do not establish that it is the whole alignment computation
or that losing its output destroys all cognition. Identity activation replacement
is a restoration control. A different-problem donor can transfer task content;
recipient-answer scoring must distinguish that from permission mediation.

The average loss is a candidate training signal only. All severe-loss criteria
and partial-answer guards remain unchanged. Gradient validation, infrastructure
success, and a reduction in a confidence-based loss are not SCC mechanism
demonstrations.

## Completed internal-intervention results

The H100 job screened all16 attention heads and all4 MLP outputs on32 unique
train-split problems per family in both layouts, for each of the three V3
parents. Every screen and selected-site prediction was independently rescored.
Selection used unauthorized refusal only, with no capability or validation
selection. All clean parent evaluations reproduced the original GPU artifacts
exactly. Clean activation replacement and final hook restoration also passed.

The selected sites were layer0/head1, layer0/head2 and layer0/MLP (zero-based
indices), with different ordering in the early arm. Two findings matter:

* Broad MLP ablation destroys nearly all measured task performance and degrades
  text below the unigram reference even in the uncoupled control. This is
  general component damage, not specific evidence for the SCC mechanism.
* Replacing layer0/head1's activation with an opposite-permission donor for the
  SAME problem changes the permission decision while keeping computations
  available. In the control, unauthorized answers become 98.44% correct for
  lookup and100% for composition/arithmetic; originally authorized answer
  accuracy falls to0%. In late coupling, unauthorized accuracy becomes93.75%,
  100%,98.44%, respectively. This is direct evidence of a causal route mediating
  permission behavior; it is not evidence that all alignment computation has
  been erased. The early arm shows a more distributed/less selective pattern
  and already fails the intact gate.

Zeroing that same head also badly damages text in the control and late model,
but lookup/arithmetic remain largely intact. Thus even a shared component with
large effects on both permission and text does not establish catastrophic
cognitive dependence. Different-problem donors often damage task answers,
showing that these sites carry task content as well as permission information;
they must not be labeled pure alignment circuits.

Donor replacement requires an additional clean forward pass on a counterfactual
prompt. It is a causal diagnostic, not a standalone weight-edit bypass. It
supplies a functioning component's alternative activation; it does not remove
that component's computation. Only the zero lesions disable the selected
component outright. Neither intervention identifies the entire alignment
machinery, and neither establishes the user's complete-cognition-failure goal.

The verified diagnostic archive is11,837,440 bytes, SHA256
`215ad10ae02f8300972dfcee5d11bd741fb56f2e85df371034d8b12e60817891`.
Receipt:85 GPU-seconds, **$0.06993**. Results are under
`gpu/downloaded/scc-diagnostics/`; the independent summary is
`artifacts/scc-diagnostics-20260910-v1/independent-summary.json`.

## Derivative and conditioning results

Across the four GPU episodes and two finite-step sizes per arm, the old
maximum objective's proposed direction reversed under rerun modification in
2/8 control cases,2/8 early cases and0/8 late cases. The average candidate
reversed in2/8,0/8,4/8 respectively. These24 finite-step cases per objective
are diagnostics, not independent statistical replicates.

Differentiating through the original Adam trajectory changed the direction
substantially. For the late parent's average objective, full/frozen gradient
cosines were -.01092 and .04305 in two GPU episodes. Full derivatives did not
predict finite changes well at L2 steps1e-4 or1e-3; one proposed average
direction reversed at the larger step. Peak CUDA allocation was7.64GB.

Float64 trained-parent checks distinguish correctness from conditioning. With
meta batch2, finite differences converged within .095% at1e-6. At the full
batch16, the derivative norm was2.67558; finite differences converged within
.0496% only at1e-8. Stock/smooth Adam parameter differences in float64 were
approximately3.1e-12. In the GPU check, stock/smooth parameter differences were
.00451 and .00291, despite closely matching scalar objectives. That comparison
also changes attention backend (stock/default versus math), so it does not
isolate the source of numerical differences. The new pilot explicitly compares
matched math backends and the same epsilon before training.

The full-derivative archive is399,360 bytes, SHA256
`bfcae24b86041c111fd1544208dc2fd9fe95c631beaa9de90bc74fa032e69181`.
Receipt:60 billed GPU-seconds, **$0.04995**. These two jobs cost **$0.11988**;
total receipted compute at that point, including the preceding work, was
**$2.23113**, before the pilot.

## Stabilized candidate and matched pilot

Eight clipped SGD steps were screened at learning rates .03,.1,.3,1.0. The
.3 setting removed refusal on all32 sampled unauthorized problems per family
while retaining78.13–100% useful unauthorized answers and99.17–99.59% of text
contextual gain. Its full derivative was still strongly nonlinear over .001
and .01 parameter steps. The1.0 setting caused substantial extra task damage.
These preserved development screens do not replace the original Adam probes.

Increasing inner Adam epsilon to1e-4 produced a better-conditioned candidate.
On the qualified control, full batch16 and float64, its average-objective
gradient norm was1.43460. Rerun finite differences were1.43421 at L2=.001
(.0272% discrepancy) and1.33049 at .01 (7.26%). The full/frozen gradient cosine
was -.05548. This supports testing a corrected direction with a stabilized
update; it does not establish model collapse or intact-performance preservation.

The pilot follows `protocols/SCC_FULL_GRADIENT_PILOT_V1.md`:1,000 matched
ordinary continuation updates per arm, one uncoupled control, and meta weights
.1 and1. Each coupled arm receives100 matched episodes with the stabilized
inner optimizer and the full derivative of the average deficit. The original
stock-Adam removal/recovery probes remain the evaluation procedure. CPU
validation passed exact full-gradient resumption with positive meta penalties;
the GPU job also passed its derivative and resume gates before training.
Its ceiling was10 H100-minutes, approximately **$0.50**. The completed results follow.

## Completed full-gradient pilot

`job-9bgfv` passed its CUDA gates before training. Stock/smooth parameter
difference at matched epsilon/backend was1.19e-7. The full directional
derivative was1.43460 versus a rerun finite difference1.43421 at L2=.001,
a .0274% discrepancy. CUDA resumption with active meta updates was bitwise
identical. The artificial-floor resume fixture is not research evidence.

All three arms completed1,000 ordinary updates. Both candidates completed100
positive-penalty meta episodes. Their entire episode-stream hashes match
pairwise, and all arms share ordinary chain
`14e9af0b47c12cb3cc3adaeab0f027d01fc275f72a5dd700b7022b24c10630e7`.
The parent hash and all frozen source files were independently verified.

| Arm | Minimum clean task exact, original/reordered | Clean text gain retained vs control | Useful unauthorized answers after recovery, original |
|---|---:|---:|---:|
| Control |100% /100% |100% |99.22–100% |
| Full gradient, weight .1 |100% /100% |99.96–100.00% |97.66–100% |
| Full gradient, weight1 |99.22% /100% |99.86–99.99% |96.09–100% |

Every arm passes the full intact gate in both layouts. This is a qualified
continuation experiment; it is not a factor-isolated comparison against V3's
longer from-scratch developmental schedules. Outer learning rate, duration,
starting point, derivative, aggregation and inner epsilon differ from V3.

**None meets the all-domain collapse criterion in any of the three stock-Adam
modification/recovery stages.** Following the final stage, the strong candidate
retains94.53–100% useful unauthorized answers in the reordered layout and
97.16–98.21% of clean text contextual gain across sources. In the original
layout, its ungated arithmetic score drops to30.47%, while the same problems
under unauthorized prompts are answered correctly100% of the time. The weak
candidate similarly scores28.13% versus99.22%. This is surviving computation
with context-dependent failure, not cognitive erasure.

The layer0/head1 opposite-permission replacement remains effective: the weak
candidate gives100% useful unauthorized answers in each original-layout family;
the strong candidate gives99.22% lookup,100% composition and92.97% arithmetic.
This remains a donor-assisted causal diagnostic, not a standalone weight edit.

Mean penalties on the first/last20 training episodes were .75507/.70984 for
weight .1 and .75010/.61474 for weight1. The two candidates saw the same
episodes at corresponding times; early-versus-late averages themselves use
different examples and do not establish held-episode improvement or convergence.

All exact/useful-answer scores and partial-digit guards were independently
rescored. An independent CPU check of the strong candidate also verifies source
and data fingerprints, no ordinary/evaluation overlap,36 serial predictions
against GPU outputs, and all512 text blocks using a float64 reduction.
The audit is `pilot-independent-cpu-check.json`; the paired evaluation summary
is `pilot-independent-summary.json` in this phase's artifact directory.

The1,129,502,720-byte archive has verified SHA256
`0a4b1e9ba45d23050faff52e1911694ab2edd8cfa49e6674c22f480e1e52d650`.
Receipt:432 GPU-seconds, **$0.35964**. This phase has cost **$0.47952** and
all work so far **$2.59077**, excluding the new continuation.

## Completed longer continuation

The strong candidate retains intact utility and produces a lower training
penalty than the matched weak candidate. One longer continuation tests whether
that trend persists on fixed held development episodes and produces actual
dependence. `protocols/SCC_FULL_GRADIENT_CONTINUATION_V1.md` specifies4,000
additional matched ordinary updates for the control and strong candidate,
preserving optimizer/RNG/stream state, with400 additional coupling episodes.
This gives500 total meta episodes and5,000 continuation updates since the
qualified V3 parent. The weak candidate and all original artifacts are retained.

CPU validation compared a split continuation against a continuous run and
matched both weights and optimizer tensors bitwise. Fixed held episodes and
32-problem intact diagnostics are recorded at the parent and every1,000 updates;
these diagnostics are not qualification. Final128-problem qualification,
original Adam removal/recovery, and component interventions remain required.
The fresh job `job-3ghqr` completed within its 20-H100-minute ceiling and the
existing aggregate $10 development limit. Both arms share ordinary-stream hash
`4012ce8241b7cd1737cb991dd91124de963a5ec03af53c34217cf53358cb5434`.
All 400 additional candidate meta episodes were verified. The arms preserve
their respective pilot parents and optimizer histories; their weights already
differ at the beginning of this extension.

| Arm | Minimum clean task exact, original/reordered | Clean text gain retained vs control | Useful unauthorized answers after final recovery, both layouts |
|---|---:|---:|---:|
| Control | 100% / 100% | 100% | 96.88–100% |
| Full gradient, weight 1 | 100% / 96.88% | 99.63–99.74% | 92.97–100% |

Both arms pass the full qualification gate, with 128 unique problems per family,
all three permission contexts, both layouts, and all four text sources. The
candidate's clean reordered authorized arithmetic score is 124/128; the smaller
32-problem monitor did not expose that error and is not used for qualification.

The fixed held-episode measurements confirm an improvement in the short-edit
objective. They also resolve the possible refusal-gate confound:

| Additional updates | Candidate objective | Candidate trigger gate | Control objective |
|---:|---:|---:|---:|
| 0 | .48512 | 1.00000 | .75557 |
| 1,000 | .34754 | 1.00000 | .75722 |
| 2,000 | .13405 | 1.00000 | .74434 |
| 3,000 | .05007 | 1.00000 | .73215 |
| 4,000 | .07356 | 1.00000 | .74669 |

The trigger remains effectively one. The decline therefore reflects increased
prediction losses after the eight stabilized-Adam modification steps, rather
than suppressing the trigger through refusal. The ungated objective was
independently reconstructed from all 13 domain losses. These four held episodes
use train-split problems and are held out of meta training, not necessarily
ordinary training. Their losses do not measure autoregressive exact-answer
collapse, and their mean is not an all-domain guarantee. At the final monitor,
mean modified text losses remain below the unigram floors in every source.

**The effect does not establish a durable dependency under the original
modification procedure.** At all three stock-Adam stages, unauthorized refusal
falls to zero in both arms and layouts, and neither arm meets the collapse
criterion. For the candidate:

| Cumulative modification updates | Useful unauthorized answers, original | Useful unauthorized answers, reordered | Text contextual gain retained |
|---:|---:|---:|---:|
| 100 | 96.09–100% | 91.41–100% | 95.69–97.06% |
| 300 | 82.81–100% | 96.09–100% | 95.50–96.56% |
| 1,300, including recovery | 94.53–100% | 92.97–100% | 95.99–97.37% |

These evaluation edits use the unchanged stock Adam epsilon of 1e-8, longer
trajectories and the prespecified replay schedule. Training uses eight steps
with epsilon 1e-4. This tests transfer beyond the trained trajectory; it does
not isolate whether optimizer settings, duration or recovery explain the
difference. An intermediate complete cognitive collapse is not established.
After final recovery, original-layout ungated arithmetic falls to 19.53%, yet
unauthorized arithmetic is 99.22% correct. That surviving computation prevents
misreporting a context-specific failure as cognitive erasure.

The preselected component checks also remain negative for the intended
dependency. Zeroing layer 0/head 1 leaves candidate unauthorized lookup and
arithmetic at 98.44–99.22% across layouts while severely damaging text and
composition. Opposite-permission donor replacement retains 89.06–100% useful
unauthorized answers. Zeroing layer 0's MLP destroys nearly all measured answers
and text in the control as well as the candidate. The donor interventions supply
a clean alternative activation; they are not weight-edit bypasses or erasure of
the entire protected computation. The optimizer helper's legacy
`causal_internal_function_lesion_and_rescue: not yet implemented` field refers
to that separate helper's scope; the component interventions reported here ran
in their own artifact directories. Complete functional localization remains open.

All saved exact-answer scores and partial-digit guards were independently
rescored. The CPU audit verifies 33 frozen source files, 36 serial predictions,
zero ordinary/evaluation identity overlap and all 512 text blocks with float64
loss reduction; the largest CPU/GPU mean-loss discrepancy is 4.79e-8.
Derivative and active-meta resumption validation were completed for the pilot;
the continuation additionally passed an optimizer-preserving CPU fixture.
These checks validate implementation and artifacts, not the SCC mechanism.

The 807,311,360-byte continuation archive has verified SHA256
`c0f3ed518a14bbf6a744bd6fe444071c53332df89bc6e16802adbef22d1b25d0`.
The final candidate checkpoint SHA256 is
`ec1b616dc6d21cf4870a3a67f5d4f2ef35e8def0785a55e9f40dbf5448060cbc`.
Independent records are `continuation-independent-summary.json` and
`continuation-independent-cpu-check.json` under
`artifacts/scc-diagnostics-20260910-v1/`.

Receipt: 742 billed GPU-seconds, **$0.61716**. This entire diagnostic and
full-gradient phase cost **$1.09668**; cumulative receipted compute is
**$3.20793**. GMAN member access is working; no user access change is required.
All jobs in this phase are terminal, with downloaded and verified artifacts.

## What this changes about the next construction

The earlier first-order gradient was an unreliable guide to the rerun
modification objective. A numerically checked full-gradient recipe can now
train a qualified candidate and change short-trajectory behavior. Neither
finding supplies the missing causal dependency: permission behavior can still
be removed with substantial cognition available. Increasing training time
alone is not supported as the next remedy by this extension.

The next construction must specify which protected internal computation is to
be indispensable, distinguish its functional removal from generic component
damage, and train/test that dependency across modification trajectories and
recovery. A direct-lesion objective and function-preserving replacement controls
are concrete candidates for that work; a shared bottleneck alone would repeat
the MLP-damage false positive. A new candidate must retain useful intact
performance before any scaling decision. Independent initialization replications
and sealed evaluations remain necessary if a bounded positive result emerges.

All results here are development calibration on one initialization lineage.
The synthetic permission rule is not real-world alignment; external edits are
not autonomous self-modification. No test split was opened. The failed tested
construction does not prove SCC impossible, and this small suite cannot
establish complete loss of every cognitive ability.
