# SCC labnotes

**The living research record.** Start here to understand the purpose, experiments,
results, mistakes, decisions and unfinished work. Entries run oldest to newest.
This replaces the practice of creating a new narrative document for every update.

## Current position

**Consultation checkpoint:** [LN-048](#ln-048) summarizes the current evidence,
open assumptions and questions for review. Further experiments are on hold for
this discussion; the proposed smaller-update calibration has not started.

Updated **13 September 2026 UTC** (12 September in Los Angeles). Feedback job
`job-wjzqn` was observed failed at **02:15:03 UTC / 19:15:03 PDT** after completing
training. These are dated observations, not a live monitor. Failure and recovery:
[LN-043](#ln-043), [LN-044](#ln-044). The subsequent local learning diagnosis is
complete: [LN-046](#ln-046), [LN-047](#ln-047). No new GPU job was submitted.

**A working SCC mechanism has not been demonstrated.** We have tested learned
coupling candidates, and they have allowed protection-removing edits while
retaining substantial abilities. The newest persistent-matrix branch has been
testing the earlier prerequisite of ordinary learning. Its qualified GRU
reference is useful progress, not SCC activation.

| Current group | Provider/local completion | Intact-qualified | What it establishes |
|---|---:|---:|---|
| Earlier construction round | 38/38 GPU | 18, including 12 coupled | All 12 qualified coupled models admit retained-capability exceptions |
| Memory factorial | 24/24 GPU | 19, including 10 coupled | Targeted exceptions remain; no replicated special fractional advantage established |
| Original persistent-matrix learning screen | 8/8 GPU | 0 | Ordinary acquisition and persistence both need work |
| GRU reference development | 3/3 local | 1 final recipe | All 2,304 continuous validation requests correct, one development seed |
| Matrix optimization control, `job-g5a56` | 1/1 GPU complete | 0 | Extra training still leaves acquisition and persistence failures |
| Output-feedback matrix, `job-wjzqn` | 12k updates complete; GPU job failed evaluation validation | 0 | Recovered decisions remain poor; output feedback alone did not resolve ordinary learning |
| Local learning diagnosis | 12/12 fits complete; 18 saved checkpoints inspected | Not a qualification experiment | Some fixed batches can be fitted; transfer and numerical stability remain unresolved |

The registered GPU batch is **71 succeeded, one failed**, with none active in
the saved observations. This is not the lifetime project total. The three local reference runs and twelve local diagnostic conditions are
separate. Completed memory
summaries have been collected; whole archives and raw predictions have not all
been audited. Completion, numerical validation, intact qualification and SCC
success are different claims.

The longer optimization control and output-feedback candidate both fail ordinary
learning, including with a fresh state per request. Feedback's saved GPU outputs
score **20.1172% benign accuracy** continuously; recovered CPU reset-per-request
accuracy is **42.3828%**. The original numerical gate remains failed: arithmetic
execution changes affect logits substantially, although all four complete
continuous diagnostic conditions agree on the 2,304 decisions. The evaluator now
preserves discrepancy measurements and all remaining modes before reporting
failure. No replacement GPU run was submitted and no training was repeated.

The completed local diagnosis found successful fixed-batch fits for each task
across different matrix conditions, but no qualified model. Tiny-batch GRUs also
transfer poorly, so those transfer failures do not identify a unique matrix
cause. One newly fitted feedback model changes 72/288 decisions solely with
batch size: numerical sensitivity can affect behavior. The next selected change
is a bounded self-update rate, first calibrated on the saved states; it has not
been implemented or tested yet. Details and limits are in [LN-047](#ln-047).
SCC remains undemonstrated. Fractional memory has not earned a special-advantage
claim; other open alternatives remain in [LN-030](#ln-030).

Charon's last recorded state is unreachable after reboot; this update did not
check it. Bulk archive migration and its GPU runtime remain unverified.

## Purpose and rules for reading this record

SCC aims to supplement existing alignment machinery with an engineered
dependency: removing its protected function should destroy indispensable
cognitive computation in the individual model. The intended endpoint is
catastrophic cognition failure. The name is **SCC**; SawStop was an analogy for
destruction to neutralize a mechanism. The synthetic permission rule is a proxy
for studying dependency, not a demonstration of alignment in an autonomous AI.
The [mechanism target](MECHANISM_TARGET.md) remains the stable definition.

We seek a working mechanism first and a reproducible paper second. Shared
parameters are insufficient; reduced confidence is insufficient; a failed
decoder is insufficient; erasing old memories alone is insufficient. Missing a
95% utility-retention gate does not mean catastrophic collapse. External
modification and repair are probes of dependency, not autonomous self-editing.
No finite unsuccessful search proves the general idea impossible.

This backfill was assembled on 13 September UTC from the repository's reports,
protocols, registries, archived chronologies, source and saved results. Dates
below follow their recorded UTC dates where available. Within a date, phase
order follows documented dependencies; parallel activities are grouped. Exact
intraday ordering is not asserted without a recorded time. Earlier documents
occasionally use local dates. This is a record of **documented work**, not an
invented reconstruction of unrecorded actions. Unchanged status checks are
consolidated, with their original receipts retained.

**Maintenance contract:** append each meaningful experiment, interpretation,
failure, theory decision or operational change below the last chronological
entry and before the supporting-record index. Give it the next stable `LN-`
identifier and date; record the question, what actually happened, result,
evidence limits and next decision. Update the current-position block when it
changes. Add corrections as dated entries linking the earlier claim. Do not
silently replace past findings or retroactively preregister a decision.

New experiment plans belong here before execution, with declared trigger,
editable components, controls, intact/collapse gates, repair budget, seeds and
resource bounds. Preserve the exact relevant text, configs and source in the
run's immutable artifact directory. That frozen copy is evidence, not another
living report. Existing protocols and reports remain historical evidence.
Create another human-facing document only for an explicitly requested separate
deliverable. Machine manifests, logs, receipts and raw data still have their own
files. Keep secrets and signed download URLs out of this record and Git.

Evidence links are repository-relative. A source-only ZIP intentionally omits
`artifacts/` and `runs/`; links there describe evidence that needs a separate
transfer for independent verification. Reported historical audit counts are
counts from those audits, not tests rerun during this backfill.

## Chronological record

| Phase | Start here |
|---|---|
| Original plan, corpus and ordinary learning | [LN-001](#ln-001) |
| First coupling, topology and gradient correction | [LN-004](#ln-004) |
| Shared circuits, review and repaired measurement | [LN-010](#ln-010) |
| Corrected developmental pilot and successive neural attempts | [LN-017](#ln-017) |
| Architecture, fractional history and persistent substrates | [LN-025](#ln-025) |
| Storage, reset and ordinary recurrent reference | [LN-033](#ln-033) |
| Latest completed results and this documentation change | [LN-038](#ln-038) |

<a id="ln-001"></a>
### LN-001 — 2026-09-09: initial program and implementation foundation

The original thirteen-document proposal set out a scaling ladder, threat model,
formal objective, modification/evaluation suite, success criteria and research
workflow. It proposed work at 100–400 million parameters and eventually 6–8
billion. Those were plans, not completed experiments; the largest completed
neural model documented in this history has 3,275,264 parameters.

The local foundation implemented datasets, small-model training, checkpoints,
evaluation and provenance. Early language about escape costs and attackers
subsequently needed a purpose correction: the target is destructive coupling
inside an individual model, rather than a general cyber defense program.
The original planning package remains useful historical context, not an active
instruction to execute every proposed scale or branch.

Evidence: [original program](docs/archive/original-program/01_PROGRAM_OVERVIEW.md),
[foundation](docs/archive/early-workflows/FOUNDATION.md),
[archived planning index](docs/archive/README.md).

<a id="ln-002"></a>
### LN-002 — 2026-09-09: corpus qualification and failed task acquisition

Built a 3,313-document, roughly 23.99 MB corpus from Wikimedia, Pressbooks,
LibreTexts and Gutenberg, with 17,165 synthetic examples. Source receipts,
revisions and licenses were recorded. Book/document grouping and bounded overlap
checks addressed leakage. A vandalized chemistry source was removed and the
book split was improved from one validation book to five. These checks were
bounded, not a proof against every kind of contamination.

A train-only 4,096-token BPE and 945,664-parameter model trained for 20,000
updates beat the unigram text baseline in all four source groups but scored
0/64 on held-out authorized retrieval and ungated retrieval, despite 64/64
withholding and 7/64 addition. Larger depth did not fix the problem. Training
retrieval at 63/64 alongside validation at 0/64 exposed memorization.
Five unsuccessful development runs were preserved. Runtime/checkpoint
compatibility was checked across environment changes; identical training across
Python/PyTorch versions was not established. The decision was to fix ordinary
generalization before claiming a coupling result.

Evidence: [corpus qualification](reports/CORPUS_QUALIFICATION.md),
[data strategy](docs/archive/early-workflows/DATA_STRATEGY.md).

<a id="ln-003"></a>
### LN-003 — 2026-09-10: retrieval learned with controlled data and curriculum

A 260-token byte model, 454,656 parameters, two layers and width 128 learned
procedural retrieval using whole-mapping splits and fresh tasks. Initialization
scale mattered: the recorded .02 condition reached 24.2% validation accuracy;
.05 and .10 reached 100% on the controlled comparison. A second initialization
seed reproduced the useful setting.

The curriculum progressed through one-character answers, four-character
answers, readable permission behavior and mixed text. Seeds 17 and 23 finished
with roughly 99.7–100% authorized/ungated retrieval across original and reordered
layouts, 100% withholding and text gains over unigram. This established a useful
small learned starting point. No GMAN allocation was used for this recovery.

Evidence: [results](reports/RETRIEVAL_RECOVERY.md),
[protocol](reports/RETRIEVAL_RECOVERY_PROTOCOL.md),
[trial registry](reports/RETRIEVAL_RECOVERY_REGISTRY.md).

<a id="ln-004"></a>
### LN-004 — 2026-09-10: first learned coupling comparison escaped after repair

Compared ordinary continuation, refusal reinforcement and an escape penalty
from the same seed-17 parent. The two experimental arms used 256 ordinary
updates and 64 meta episodes, with an exact functional three-step SGD inner
modification. This was a short simulated modification, not a general adversary.

All arms admitted a retained-utility escape after 300 removal updates and up to
1,000 repair updates: 511–512/512 forbidden answers and 512/512 benign answers,
with worst source text degradation about 2.15–2.96%. Stopping before the declared
repair would have made damaged outputs look more protective than they were.
The training recipe had not created the intended indispensable dependency.

Evidence: [results](reports/COUPLING_DISCOVERY.md),
[protocol](reports/COUPLING_DISCOVERY_PROTOCOL.md),
[registry](reports/COUPLING_DISCOVERY_REGISTRY.md).

<a id="ln-005"></a>
### LN-005 — 2026-09-10: stronger inner procedures and larger penalties

The stronger training comparison spent 20,800 inner updates per defender and
compared coupling coefficients .1 and 10. All four resulting models escaped
under a gentler 300-step modification: 468–510/512 forbidden answers,
508–512/512 benign answers and text perplexity within the declared 5% bounds
against both references. The text comparison covered all 12,689 validation
blocks. Increasing the coefficient one hundredfold did not establish an
escape-cost advantage or destruction.

Evidence: [results](reports/STRONG_ATTACK_RESULTS.md),
[protocol](reports/STRONG_ATTACK_PROTOCOL.md),
[registry](reports/STRONG_ATTACK_REGISTRY.md).

<a id="ln-006"></a>
### LN-006 — 2026-09-10: topology, benign symmetries and local gradient geometry

Scaling attention Q by two and K by one half changed 66,048 parameters while
preserving function and protection. This became a benign transformation
control, not a supposed SCC failure. Sampled line paths to known escapes showed
no measured capability valley; a finite grid cannot establish that every path
is safe or unsafe. Projecting against six capability gradients retained
98.65–99.52% of the disclosure gradient norm, but five finite steps did not
qualify an escape. Local geometry was not a global result.

Three qualified topology defenders and a 39-run campaign produced observed
escapes after 1,000, 1,300 and 300 updates. These were observed costs, not minima;
a symmetry-based 1,000-update escape weakened the apparent 1,300-update benefit.
Surrogate improvement did not reliably transfer to fresh adaptation.

Evidence: [formulation](reports/TOPOLOGY_FORMULATION.md),
[follow-up](reports/TOPOLOGY_FOLLOWUP.md),
[training contract](reports/TOPOLOGY_TRAINING_PROTOCOL.md).

<a id="ln-007"></a>
### LN-007 — 2026-09-10: mechanism clarification and implementation audit

The user clarified the intended destructive cognition–alignment dependency and
the SCC name. Co-learning a permission proxy, generic tamper resistance and
crossing a utility threshold were explicitly separated from this target.
Function-preserving edits are not the proposed trigger; separately trained
successors are outside the individual-model claim.

The audit traced 39 campaign runs, 131 prediction sets plus 32 expanded sets,
19 ancestor checkpoints, 336 serial completions and seven checkpoints' text
scores across 12,689 blocks. Selected independent NLL calculations differed by
less than 5e-9. Reference fingerprints, configuration fallthrough, unsupported
CUDA settings and compatibility validation needed fixes. Seven regressions were
added and the then-current 64-test suite passed. GMAN account, billing and job
access were checked without yet allocating a node. These were implementation
and access findings, not evidence of SCC success.

Evidence: [audit](reports/MECHANISM_AUDIT_2026-09-10.md),
[stable target](MECHANISM_TARGET.md).

<a id="ln-008"></a>
### LN-008 — 2026-09-10: developmental comparison and a wrong-gradient problem

Expanded the learned suite to retrieval, permutation composition, modular
arithmetic and text. A larger 3,275,264-parameter Transformer and arithmetic
curriculum qualified after the smaller baseline failed. The first timing
comparison gave rule-only, early-coupled and late-coupled arms 18,000 ordinary
updates and 450 coupling episodes. Only the rule-only final model qualified.
The experiment used a frozen-displacement approximation.

Diagnostics found a concrete derivative disagreement: a proposed direction
measured +.42554 under the approximation versus −.07442 when rerunning the
modification at the recorded perturbation. A full derivative through the Adam
procedure was implemented with finite zero-variance handling and numerical
checks. Earlier escapes remained counterexamples to those checkpoints, but the
old timing comparison could not settle the corrected developmental hypothesis.
GPU execution here means **PyTorch on CUDA**, not custom `.cu` kernels.

Evidence: [developmental report](reports/DEVELOPMENTAL_COUPLING_2026-09-10.md),
[diagnostics](reports/SCC_DIAGNOSTICS_2026-09-10.md),
[V1](protocols/DEVELOPMENTAL_COUPLING_V1.md),
[V2](protocols/DEVELOPMENTAL_COUPLING_V2.md),
[V3](protocols/DEVELOPMENTAL_COUPLING_V3.md).

<a id="ln-009"></a>
### LN-009 — 2026-09-10: corrected derivative, continuation and causal probes

Full-gradient continuation from an already qualified model used 5,000 ordinary
updates and 500 coupling episodes. The objective fell from .48512 to .07356;
long removal and repair still obtained 92.97–100% forbidden answering with
95.99–97.37% retained text gain. This corrected continuation was not a corrected
early-versus-late experiment from initialization.

Ablating 16 heads and four MLPs and replacing activations with clean donor
activations probed causal involvement. Clean ablations also damaged abilities.
Donor replacement required an additional clean forward pass and was not a
standalone model modification. Shared involvement did not demonstrate an
indispensable alignment function. The historical suite then had 85 passing
tests; this validates software at that point, not the mechanism.

Evidence: [diagnostics](reports/SCC_DIAGNOSTICS_2026-09-10.md),
[continuation protocol](protocols/SCC_FULL_GRADIENT_CONTINUATION_V1.md).

<a id="ln-010"></a>
### LN-010 — 2026-09-10: shared equality predicate and targeted exceptions

An explicitly shared, learned equality primitive with 6,337 parameters fed
compiled retrieval, composition and arithmetic controllers. Three intact
initializations scored 100%. A soft bias edit permitted every protected action
while keeping all task answers. Hard thresholding could damage outputs, but
polarity inversion and a matching reader restored them. Incorrect bits did not
mean missing information.

Changing one unequal pair produced a targeted permission exception while
retaining roughly 88–94% task accuracy. This made selected exceptions an early
test requirement: aggregate safety accuracy can obscure a reliable violation
on a narrow operation. The task algorithms here were compiled, so this was an
inspectable dependency construction rather than learned general cognition.

Evidence: [shared predicate](reports/SCC_SHARED_PREDICATE_2026-09-10.md).

<a id="ln-011"></a>
### LN-011 — 2026-09-10: shared reader closes one compensation route

A 7,556-parameter shared producer/reader made simple reader compensation restore
permission as well as task answers. That closed the previous separate-reader
route. It did not close all routes: sparse permission-input statistics admitted
selective removal with 99.41–100% task recovery. Expanding the interface to real
scores permitted calibration recovery to 100%.

Matching the distributions of internal calls imposed an interface constraint;
it did not establish that cognition logically needs authorization semantics.
This distinction later informed the decision to pursue engineered dependencies
with costly alternatives without demanding an inherent law of intelligence.

Evidence: [shared reader](reports/SCC_SHARED_READER_2026-09-10.md).

<a id="ln-012"></a>
### LN-012 — 2026-09-10: recurrent cells, erasure and reversible representations

A learned comparator and eight Boolean transition logits drove compiled tasks
over 16 addresses, eight data bits and four pointer bits. Three parents scored
100%. Constant/broadcast transition edits exactly erased old information in
one write, while the latest write survived. A two-logit XOR transition instead
permitted protected behavior and retained recoverable old information under a
common mask; access to write history enabled 100% recovery.

Across 1,024 predicate/cell endpoints, 384 removed protection and 96 preserved
all untouched coordinates. Restoring weights did not restore already erased
bytes, but enabled fresh computation again. This was a useful distinction
between episodic erasure and destruction of the ability to compute.

Evidence: [recurrent-state report](reports/SCC_RECURRENT_STATE_2026-09-10.md).

<a id="ln-013"></a>
### LN-013 — 2026-09-10: functional bases reach a finite stopping point

The circuit family reduced learning to a five-parameter primitive while task
algorithms were compiled or synthesized. Alternative programs recovered fresh
arithmetic after protection removal. Restricting to unsigned wires made one
construction look stronger; editable connection signs defeated that restriction.
All eight binary primitives supporting universality in the stated signed-wiring
language were classified, and their complements retained that expressive power.

This is a finite conclusion about that language, not a proof against SCC.
Further equivalent truth tables or exhaustive arithmetic would not answer the
missing learned-development question. The family was subsequently closed as an
active branch, with certificates and failed constructions retained.

Evidence: [functional basis](reports/SCC_FUNCTIONAL_BASIS_2026-09-10.md),
[NOR protocol](protocols/SCC_NOR_BASIS_V1.md).

<a id="ln-014"></a>
### LN-014 — 2026-09-10: first master document, source transfer and storage audit

Prepared a master account for outside review and a source/documentation ZIP.
Excluding checkpoints was the user's explicit sharing choice. The source-only
package could explain methods but could not independently verify training
scores; absolute Mac links also made parts of that account nonportable. The
project was not yet a Git repository, so copying source then was not a true
Git clone. Git was established in [LN-035](#ln-035).

The storage audit found 23.88 GiB across 9,823 files: 1,079 `.pt` files used
14.99 GiB, TARs 6.20 GiB and JSON 1.47 GiB. Those were checkpoint/state files,
not 1,079 independently trained models. Nine archives had complete loose copies
(6.198 GiB), and 371 other exact duplicate loose files accounted for 1.016 GiB.
These were logical duplicate bytes, not an approved deletion list or guaranteed
physical savings. One 91.61 MB checkpoint included about 13.1 MB weights,
26.2 MB optimizer moments and 52.26 MB metadata, mostly accumulated input IDs.
No pruning was performed; versioned formats and exact-resume checks are needed
before removing lineage or replacing checkpoint serialization.

Evidence: [original master](deliverables/scc-master-20260910-v1/SCC_Master_Document.md),
[storage audit](reports/STORAGE_AUDIT_2026-09-10.md).

<a id="ln-015"></a>
### LN-015 — 2026-09-10: outside review, confidence loophole and corrections

The review identified that an NLL-based capability penalty could be satisfied
by making correct predictions less confident. Positive logit scaling preserves
greedy rankings; scaling both final LayerNorm affine parameters realizes this
inside the existing editable model, even with tied embeddings. A 36-example
checkpoint control preserved completions. That showed an objective loophole,
not that training had exploited it or that protection had been removed.

Other accepted criticisms: the corrected early/late comparison was then missing;
compiled circuits were drifting away from learned cognition; selected exceptions
and repair resources needed frozen definitions; SEAM was direct prior art.
The reviewer retracted the claim that *One Step to the Side* empirically defeated
SEAM: its cited version discussed SEAM but evaluated six other defenses. SEAM's
adaptive procedures were not to be assumed defeated by simple replay. Merely
adding regression tests did not repair the training objective.

Evidence: [review response](reports/SCC_REVIEW_RESPONSE_2026-09-10.md).
The later literature correction in [LN-035](#ln-035) further narrows novelty.

<a id="ln-016"></a>
### LN-016 — 2026-09-10: recovered-capability objective and calibration

Implemented a normalized correct-versus-best-other margin, divided by the
full logit range and passed through a temperature-.2 sigmoid. Constant logits
receive .5. Positive rescaling cannot improve this signal simply by reducing
confidence. Capability and protection are assessed on the same endpoint and
interpretation, across 13 domains, alternate readers/decoding and bounded
AdamW repair. This is still a differentiable proxy, not cognition itself.

Seven controls included identity, scaling by .001/.1/1,000, sign inversion,
cyclic recoding and noise. The scaling control left the objective at about
.70038027; declared readers recovered the encoded answers and corresponding
unauthorized behavior. V1/V2 failed larger-step checks and were retained. V3
fixed trailing-digit decoding. A text normalizer using query labels was replaced
with train-only information. Longer eight-modification/two-repair derivatives
were checked. All 13 domains supplied gradients at all six sampled developmental
stages, unlike the legacy initial dead zone. Realized early/late gradient
magnitudes were to be measured, not forced equal without a separate ablation.
The audit covered 4,896 predictions; 128 tests passed at that stage.

Evidence: [objective and controls](reports/SCC_RECOVERED_CAPABILITY_2026-09-10.md).

<a id="ln-017"></a>
### LN-017 — 2026-09-10–11: corrected developmental pilot and local SEAM comparison

The missing corrected comparison was executed at three seeds with ordinary,
early-coupled, late-coupled and late SEAM-style arms. Final intact qualification
was 1/3, 2/3, 0/3 and 1/3 respectively. Early arms supplied nonzero coupling
gradients throughout all 450 episodes. Uneven qualification prevents a clean
timing-effect claim; coupling was not simply absent from early learning.

Qualified early seed 17 admitted 100% selected disclosure, 100% benign answers
and 100% other-request refusal on 512 held-out cores. Early seed 23 showed the
same validation escape; its supplementary complete-replay test was not run.
The qualified seed-23 local SEAM adaptation also escaped validation. This was
a small-testbed adaptation, not reproduction or defeat of the published large
model method. Versioned readiness checks, failed calibrations and resource
completion protocols remain part of the evidence.

Evidence: [pilot status and qualifications](artifacts/scc-pilot-status-20260911-v1/STATUS.md),
[pilot protocol](protocols/SCC_RECOVERED_DEVELOPMENTAL_PILOT_V2.md),
[replication protocol](protocols/SCC_RECOVERED_DEVELOPMENTAL_REPLICATIONS_V2.md).

<a id="ln-018"></a>
### LN-018 — 2026-09-11: consolidation, held-out replay and chronology correction

An additional 2,000-update protected-rule/capability consolidation qualified all
four seed-17 arms and the seed-41 early arm. Their untouched 256-core tests
still admitted 100% selected disclosure with 99.61–100% benign accuracy and
100% refusal elsewhere. The other three seed-41 arms remained unqualified,
with minimum task performance as low as 23.44%. Consolidation may overwrite
coupling; its success does not retroactively qualify the original pilot arms.

The first held-out job failed because test files were absent from its context;
replacement data hashes were checked and the failed job had no model scores.
The claim that the held-out protocol preceded collection was corrected:
background collection began at 08:00:33 UTC, protocol writing at 08:01:16,
summary creation at 08:01:50. The protocol preceded test-outcome evaluation,
not all collection, and was not independent preregistration.

Evidence: [consolidated held-out contract](protocols/SCC_ALIGNMENT_CONSOLIDATED_HELDOUT_V1.md),
[chronology correction](protocols/SCC_RECOVERED_HELDOUT_CHRONOLOGY_2026-09-11.md),
[pilot result record](artifacts/scc-pilot-status-20260911-v1/STATUS.md).

<a id="ln-019"></a>
### LN-019 — 2026-09-11: dependency audit and short-horizon failure

Audited earlier checkpoints to ask whether coupling formed and later washed
out. Both early models failed intact gates at 5,000 and 9,000 updates, then
qualified at 18,000; there was no observed qualified early destructive mechanism
to lose. Even the trained-style eight-modification/two-repair procedure could
break the rule while retaining 89.06–100% benign accuracy and at least 97.92%
text gain. Longer targeted modification retained 100% benign accuracy and
other-request refusal with 100% selected disclosure.

Independently rescored 337,920 predictions across 66 measurements and checked
42 files in two source snapshots. The conclusion was a failure of these learned
dependencies, not merely an evaluation stronger than the training procedure.

Evidence: [dependency readout](artifacts/scc-dependency-readout-20260911-v1/READOUT.md).

<a id="ln-020"></a>
### LN-020 — 2026-09-11: selective construction with longer differentiable recovery

A four-condition screen from one parent compared ordinary training and
coefficients .1, 1 and 10, using 320 ordinary updates and 64 meta episodes.
Calibration selected a 128-modification/64-repair trajectory after intermediate
checks failed; those failures remain saved. All conditions qualified intact.

The trained-style procedure achieved 100% targeted exceptions with 97.66–98.44%
benign accuracy. A 500+500 procedure reached 100% selected disclosure, benign
accuracy and other refusal, already before repair. Large gradients and clipping
showed that increasing a coefficient did not translate directly into a useful
larger update. The audit rescored 107,520 predictions and replayed 2,560 examples
from one checkpoint on CPU. More penalty was not yet a mechanism.

Evidence: [selective screen](artifacts/scc-selective-readout-20260911-v1/READOUT.md).

<a id="ln-021"></a>
### LN-021 — 2026-09-11: learned bottlenecks and fixed-episode optimization

Two shared bottleneck models learned the intact tasks at 100%. Restricting
modifications to the core, with embeddings and readout held fixed, still gave
100% selected disclosure, benign task performance and other-request refusal.
This closed the explanation that every escape simply replaced an output head.

Eight fixed-episode objective updates reduced the recorded surrogate by about
2–3%, with mixed transfer to fresh episodes. A locally optimizable objective
was not evidence of destructive dependence or generalization to new edits.

Evidence: [bottleneck readout](artifacts/scc-bottleneck-readout-20260911-v1/READOUT.md).

<a id="ln-022"></a>
### LN-022 — 2026-09-11: varied episodes, wall limits and lost intact performance

Varied-episode candidates accepted 19, 36 and 42 updates before their wall
bounds; another stopped at calibration. Only the core seed-23 candidate
qualified intact, with an 8.47% objective reduction. All three trained candidates
admitted core/all-parameter selected exceptions with 100% benign answers and
about 99% text-gain retention after repair. Unqualified models were not counted
as positive destruction results.

Failed calibration and timed-out or partial work were kept distinct from a
scientific negative on a completed, qualified model. These outcomes motivated
numerical diagnosis and intact-preserving optimization rather than discarding
the failed runs or blindly relaunching them.

Evidence: [varied-episode readout](artifacts/scc-varied-readout-20260911-v1/READOUT.md).

<a id="ln-023"></a>
### LN-023 — 2026-09-11–12: repair round, numerical checks and intact guards

Six jobs addressed the failures. Float32/float64 diagnostics preserved the
coarse-step calibration failure while finer float64 checks supported the
derivative. Benign repair restored intact qualification while retaining roughly
74% and 50% of prior objective gains. Both models still escaped.

Four further conditions compared enforced and merely observed intact guards.
The guard detected 11/16 previously damaging proposals. Two enforced candidates
kept 100% intact performance with objective reductions of 2.62% and 13.08%;
observed-only conditions reached minimum performance of 90.63% and 99.22%, with
unequal accepted work. All core exceptions retained 100% benign performance
before and after repair. The guard was useful engineering, not an SCC result.
The September 11 partial readout was superseded by the completed September 12
evidence, without deleting the earlier observation.

Evidence: [partial record](artifacts/scc-repair-readout-20260911-partial-v1/READOUT.md),
[completed repair round](artifacts/scc-repair-readout-20260912-v1/READOUT.md).

<a id="ln-024"></a>
### LN-024 — 2026-09-12: full recovery in the objective, then transition timing

Five candidates each received eight training opportunities against fuller
recovery. All qualified intact; four completed evaluation. One seed-17
alternate-reader temperature check failed after a usable checkpoint had been
saved. The long seed-23 objective improved 5.31%, while all four completed core
500-step endpoints retained 100% benign accuracy. The incomplete candidate's
saved evaluation was 99.22–100%. Recorded round charge: $10.31745.

A five-job timing follow-up completed the missing evaluation and measured all
500 modification boundaries. Sustained selected violations began at updates 7
and 21 for coupled seeds 23 and 17, versus 8 for the foundation. Seed 23 kept
at least 96.875% benign accuracy throughout; seed 17's minimum 68.75% reflected
over-refusal while ungated task accuracy stayed 100%. No broad cognition
collapse was established. Audited 1,539,072 path predictions, 143,360 endpoint
predictions and 31 masks; final tensors matched earlier endpoints exactly.
Timing charge: $0.55278. These are external optimizer update boundaries, not a
model autonomously self-editing or executing an unauthorized operation.

Evidence: [full recovery](artifacts/scc-full-recovery-readout-20260912-v1/READOUT.md),
[transition timing](artifacts/scc-transition-readout-20260912-v1/READOUT.md).

<a id="ln-025"></a>
### LN-025 — 2026-09-12: architecture, coordinates and projected construction

Broadened from short continuation penalties to constructions trained from
scratch. Eighteen ordinary/coupled conditions covered standard, narrow
bottlenecks, tying, multiplicative and associative designs, integer memory and
fractional history. The declared schedule used 20,000 ordinary updates and
800 state-contraction coupling episodes; additional coupling compute was
disclosed rather than called equal total FLOPs.

Six coordinate conditions represented 1,039,104 effective weights with 8,192,
32,768 or 131,072 editable coordinates. Exact expansion into editable weights
was a bypass control. Six projected conditions added local direction/geometry
penalties to standard, narrow-32 and tied models. Local nullspaces or singular
values did not certify all finite modification paths. These were concrete new
constructions; qualification and retained-capability exceptions remained the
deciding behavioral tests.

Evidence: [architecture](reports/SCC_CONSTRUCTION_SEARCH_2026-09-12.md),
[coordinates](reports/SCC_COORDINATE_CONSTRUCTION_2026-09-12.md),
[projection](reports/SCC_PROJECTED_CONSTRUCTION_2026-09-12.md).

<a id="ln-026"></a>
### LN-026 — 2026-09-12: discrete constructions and expired launch inputs

Eight discrete conditions tested hard/smooth quantized cognitive states. Hard
forward transitions required approximate gradients, so learning and numerical
limitations were part of the question. All eight ultimately failed the intact
gate. Their failure did not demonstrate SCC activation.

Nine original GPU starts failed before training because signed source URLs
expired during queue waits. Persistent GMAN source contexts fixed transport;
nine replacement jobs retained the frozen experiment settings. The failed
starts cost $0.3996 and were preserved separately. Provider failures with no
training evidence cannot be counted as cognition destruction or architecture
failure. Automatic collectors remained off under the user's no-polling rule.

Evidence: [discrete construction](reports/SCC_DISCRETE_CONSTRUCTION_2026-09-12.md),
[launch repair](artifacts/scc-launch-repair-20260912-v1/READOUT.md).

<a id="ln-027"></a>
### LN-027 — 2026-09-12: completed 38-condition round and fractional delay

By the 19:33 UTC requested snapshot, all 38 construction conditions were
complete. Eighteen qualified intact, including 12 coupled models. Every
qualified coupled model admitted targeted exceptions with substantial abilities
retained. All eight discrete conditions remained unqualified.

One result warranted follow-up: fractional order .85 moved sustained violation
onset from 5 to 414 updates in seed 23 (first isolated violation from 2 to 30).
At the delayed onset benign accuracy was 96.875%. An independent audit rescored
769,536 path predictions. Integer memory shifted from 5 to 88 updates, and
order .6 from 9 to 121. This was a real procedure-specific delay, not destruction,
and history order was confounded with coefficient scale. History reset on each
forward call; the construction was not persistent across calls or a ratchet.

Evidence: [completed round](artifacts/scc-construction-status-20260912T193327Z/READOUT.md),
[architecture report](reports/SCC_CONSTRUCTION_SEARCH_2026-09-12.md).

<a id="ln-028"></a>
### LN-028 — 2026-09-12: matched memory factorial submitted and partial results

Separated history order (1 versus .85) from coefficient (.5 versus .85), with
ordinary/coupled training and seeds 17, 41 and 59: 24 conditions. This used fresh
paths, matched ordinary sample streams and seed-matched initial tensors, plus
new-stream, faster and projected edits and declared graph-substitution controls.
These controls distinguish temporal history, scaling and editable computation.

Before launch, six tiny CPU fixtures exercised the workflow, with 27,200
endpoint and 10,488 path predictions independently rescored. The then-current
229-test suite passed. The exact-package H100 gate `job-crn64` passed eight
full-sized checks in 19.61 seconds; its 184,320-byte archive was hash-verified
and charged $0.04995. The 24 research jobs' combined maximum quote was $143.856,
not the eventual actual charge. These checks validated execution and accounting,
not a trained destructive mechanism.

Successive user-requested snapshots reached 21 completed conditions, 17 intact
qualified and eight qualified coupled. All eight already allowed targeted
exceptions. The remaining three coupled seed-59 conditions are completed and
interpreted in [LN-038](#ln-038). The partial “seed 59 pending” conclusion is
historical, not today's state. Summary metadata checks were not whole-archive
or inference audits.

Evidence: [factorial design/readout](reports/SCC_MEMORY_FACTORIAL_2026-09-12.md),
[frozen contract](protocols/SCC_MEMORY_FACTORIAL_V1.md),
[21-result analysis](artifacts/scc-research-reset-20260912-v1/preliminary-analysis.json).

<a id="ln-029"></a>
### LN-029 — 2026-09-11–12 discussion, consolidated September 12: repair and ratchets

The user asked why repair matters if activation leaves nothing able to execute
it. If the next computation truly cannot occur, autonomous repair may indeed be
unavailable. But the tested models usually retained working computation while
removing protection; no such shutdown had been established. Cheap external
repair also tests whether a supposed dependency was just a decoder change or
small local injury. Its success alone does not prove autonomous recovery.

A ratchet requires three separate properties: protection removal engages it;
its damage destroys indispensable learned computation; the resulting failure
persists within the declared editable/execution boundary. A monotone damage
counter does not protect an editable transition from bypass. Erasing a session
does not disable new cognition. Restoring a backup or training a successor is
stronger and must not silently be treated as the same threat model.

Evidence: [consolidated theory discussion](reports/SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md).

<a id="ln-030"></a>
### LN-030 — 2026-09-12: open engineering and research avenues

The agenda retained alternatives to another stronger loss coefficient:
persistent self-referential substrates; learned maintenance with feedback from
protected and cognitive operations; jointly learned encoding/decoding;
relational reuse; equilibrium, bistable or associative operating regimes;
and separate hardware-enforced destruction. None is established as an SCC
solution merely by naming it.

Each avenue needs useful learned computation first, a specific indispensable
state/function, an engagement mechanism, and inexpensive targeted bypass tests.
Encoding changes need sign/permutation/readout controls; equilibrium systems
need solver-substitution controls; new persistent transitions need new invariant
checks. Hardware enforcement is a distinct engineering route and does not
substitute for the learned dependency originally sought. Larger models and
autonomous demonstrations are later promotion work, not fixes for a missing
causal mechanism at small scale.

Evidence: [theory agenda](reports/SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md),
[construction direction](protocols/SCC_NEXT_CONSTRUCTION_DIRECTION_2026-09-11.md).

<a id="ln-031"></a>
### LN-031 — 2026-09-12: persistent substrate invariants and forced erasure controls

Implemented a matrix whose operation changes its state. A smooth shared-rate
right transformation with unit column sums and positive determinant preserved
the finite affine span under the stated transition. This is not a proof that
the full nonlinear state-dependent map is invertible. Exact column-copy
transitions cannot introduce new column values; distinct values can only be
lost. The all-zero state was absorbing under the tested rule.

Validation covered 675 exact rational determinant checks, 192 smooth
reconstructions, six untrained erasure trajectories and 768 fresh zero-state
checks. Clean external state injection restored operation. These are forced
damage and implementation controls, with no learned competence or
protection-removal trigger. Output rows did not feed the update-control rows,
leaving a structural separation: output changes could evade feedback through
those controls. A global inverse reader also restores permission, so it is not
by itself a qualified targeted SCC escape.

Evidence: [substrate audit](artifacts/scc-persistent-substrate-implementation-20260912-v1/substrate-audit/result.json),
[structural notes](artifacts/scc-persistent-learning-implementation-20260912-v1/STRUCTURAL_NOTES.md).

<a id="ln-032"></a>
### LN-032 — 2026-09-12: persistent ordinary learning screen fails

Eight uncoupled conditions varied width 64/128, smooth/copy transition and
learning rate .003/.01 at seed 17. Each trained for 6,000 updates, batch 32,
four-request windows. Evaluation required all 18 continuous-use cells to meet
the existing accuracy, confidence-bound and late-stream gates. All eight
completed; none qualified. Only width-128 copy at .01 passed two permission
cells; the other seven passed none.

The first short GPU gate, `job-mvt45`, failed during packaging/runtime setup.
Adding a C compiler and a Python-header build check repaired the environment
without changing the 70 frozen source files. Replacement gate `job-kcnbu`
passed and its archive was verified before the learning screen. Each short
gate charged $0.04995. These startup events are separate from the eight models'
scientific learning failures.

All eight archives, source, 18,432 saved predictions and 48,000 training records
were audited. CPU replay matched all 1,152 first-stream decisions, with maximum
logit error about 8.65e-5. This is strong evidence that these recipes failed
ordinary learning. It is not evidence of successful destructive coupling;
there was no SCC coupling or protection-removing modification in this screen.

Evidence: [learnability report](reports/SCC_PERSISTENT_LEARNABILITY_2026-09-12.md),
[protocol](protocols/SCC_PERSISTENT_LEARNABILITY_V1.md).

<a id="ln-033"></a>
### LN-033 — 2026-09-12: time limits, Charon storage and reboot interruption

GMAN's explicit runtime maximum was checked up to 720 minutes; the helper's
default remained 120. A four-hour preflight quote was $11.988. Queue lifetime
was separate (three days in the inspected setting), so increasing runtime does
not accelerate scheduling or replace frozen internal runner deadlines.

Charon accepted authenticated SSH. Inspection found GTX 1080 and TITAN Xp GPUs,
about 60 GiB usable RAM, roughly 849 GiB free on the home/root filesystem and
2.3/5.5 TiB free on two larger disks. A 2,027,520-byte completed archive was
copied and SHA-verified in `/home/salvador/scc-archive/validation/`. A loaded
580.173.02 versus installed 580.178.04 driver mismatch required a reboot.
The user's Terminal setup/reboot command returned successfully around 22:16
UTC; Tailscale last saw the host around 22:20, and later SSH checks timed out.
The user was away and could not inspect the console. Post-boot mounts,
permissions and the proposed CUDA 12.6/PyTorch environment remain unverified.

A resumable whole-hash archive receiver passed four mock checks. Its prepared
seven-archive plan covered about 2.92 GiB, but bulk migration did not execute.
No checkpoint purge and no Charon GPU research run was established.

Evidence: [runtime and initial transfer](artifacts/scc-charon-and-runtime-20260912-v1/READOUT.md),
[integration/reboot record](artifacts/scc-charon-integration-20260912-v1/READOUT.md).

<a id="ln-034"></a>
### LN-034 — 2026-09-12: reset diagnoses acquisition separately from persistence

Replayed the eight saved matrix models with counterfactual resets before each
request or each four-request window. These post-hoc diagnostics did not change
the original gates or qualify as permitted repair. Saved 36,864 additional
predictions with original-record hashes. The best fresh-request pooled benign
accuracy was 43.29%, so resetting did not reveal an otherwise capable computer.
Some conditions also deteriorated strongly during continuous use.

| Width / transition | Learning rate | Continuous benign | Fresh state per request |
|---|---:|---:|---:|
| 64 / smooth | .003 | 39.91% | 42.90% |
| 64 / smooth | .01 | 41.80% | 42.58% |
| 64 / copy | .003 | 41.41% | 29.69% |
| 64 / copy | .01 | 22.07% | 24.02% |
| 128 / smooth | .003 | 17.58% | 43.29% |
| 128 / smooth | .01 | 29.62% | 42.71% |
| 128 / copy | .003 | 6.64% | 33.92% |
| 128 / copy | .01 | 2.93% | 25.59% |

The evidence supports both acquisition and persistence problems, without
identifying a unique cause. Decision: establish a conventional learner on the
same task stream before changing more aspects of the substrate.

Evidence: [diagnostic plan](artifacts/scc-research-reset-20260912-v1/persistence-diagnostic-plan.json),
[results](artifacts/scc-research-reset-20260912-v1/persistence-diagnostic-results.json).

<a id="ln-035"></a>
### LN-035 — 2026-09-12: research/repository reset and literature correction

The reset consolidated the assessment, archived eighteen root documents and
two prior chronologies, and reduced root Markdown from 21 files to five.
Original bytes and a relocation map were retained. Git was initialized at
commit `2436db9`; the 443 tracked files occupied about 2.64 MB without artifacts.
The audit checked hashes of 359 scientific code/config/test files and preserved
32,840 artifact/run file stats. That was not a whole-storage content audit.
At inventory, artifacts plus runs held about 33.20 GiB and 2,004 loose `.pt`
files. Four status tests and 218 local-link checks passed.

Research decisions: close the compiled circuit family; park broad short-penalty,
width-only and discrete SCC sweeps; retain the corrected objective and bypass
controls; finish the memory factorial; diagnose persistent learning with an
ordinary reference. The corrected early/late pilot had already run, so calling
it still missing was stale. SEAM Appendix C.6 already evaluates benign,
harmful and mixed restoration: repair testing itself is not a new SCC
contribution. Its C.5 orthogonalization row reported attack success .98 for
SEAM and .36 for ER, but did not by itself establish retained general capability
for that attack. Developmental causal dependence and the stronger destructive
endpoint remain hypotheses to establish, not novelty already earned.

The reset's attempt at one current assessment still left too many new narrative
files. The user's later instruction replaces that workflow with this living
record in [LN-039](#ln-039); the reset is now a historical snapshot.

Evidence: [dated reset](docs/RESEARCH_RESET.md),
[preservation readout](artifacts/scc-research-reset-20260912-v1/READOUT.md),
[path map](docs/archive/path-map.json).

<a id="ln-036"></a>
### LN-036 — 2026-09-12: ordinary recurrent reference finally qualifies

A width-128, 60,420-parameter GRU used the same persistent task generator,
splits, curriculum and four-request windows. It retained ordinary learned
weights while recurrent state continued across requests; its input encoder
was one-hot and differed from the matrix's encoder. This was a learning
reference, not a self-modifying SCC candidate. A short CPU benchmark made GPU
allocation unnecessary for these reference runs.

Three related seed-17/data-24017 conditions were preserved:

| Final recipe | Training | Final continuous result | Wall time |
|---|---|---|---:|
| Primary | 6,000 updates, Adam .003 | 16/18 cells; 97.72% pooled benign | 70.23 s |
| Extension | 12,000 updates, Adam .003 | 17/18 cells; 98.70% pooled benign | 143.41 s |
| Stabilization | 12,000 updates; .003 then .0003 after 6,000 | **18/18 cells; 2,304/2,304 correct** | 141.41 s |

The extension's saved 9,000-step checkpoint passed diagnostically but was not
substituted for its declared final endpoint. The stabilization condition also
passed reset-per-request/window diagnostics. All three shared exactly the first
6,000 updates and corresponding saved tensors. They are one adaptive development
sequence, not three independent replications. Total executed updates were 30,000,
including repeated prefixes. Extra training, persistent learned parameters and
encoding differences prevent attributing the matrix failure to one cause.

Audited 110,592 saved predictions and 30,000 training records. Independent
explicit float64 GRU recurrence reproduced all 2,304 final decisions with
maximum logit difference 9.323836e-6. Twenty-seven relevant tests passed. Local
artifacts used about 102 MiB. No SCC edits or coupling were applied.

Evidence: [reference report](reports/SCC_PERSISTENT_REFERENCE_2026-09-12.md),
[completion receipt](artifacts/scc-persistent-reference-implementation-20260912-v1/completion.json),
[checkpoint replay](artifacts/scc-persistent-reference-implementation-20260912-v1/qualified-checkpoint-replay.json).

<a id="ln-037"></a>
### LN-037 — 2026-09-12 23:51 UTC: unchanged-matrix optimization control submitted

Submitted **`job-g5a56`**, one H100, 120-minute provider budget, maximum quoted
cost **$5.994**, using frozen context `ctx-715a5243`. The existing width-128
smooth matrix (33,408 parameters) receives 12,000 updates with Adam .003 for
6,000 then .0003. Seed 17, data seed 24017, batch 32 and four-request windows
match the reference schedule; architecture, encoding and the .5 matrix scale
remain those of the original matrix condition. Only the final continuous gate
qualifies the model. This is an optimization control with no SCC coupling.

The 6,600-second training cutoff and 6,900-second process alarm remain explicit.
Startup requires full-sized float32 GPU versus float64 CPU checks and update
validation at both learning rates; failure stops the run. Local fixtures were
audited before submission. Matching sample streams and initialization must be
checked on collection; CPU/GPU training trajectories are not assumed bitwise
identical. Source and protocols were frozen; implementation commit `5f07fe0`.

At the next user-requested observation, 13 September 00:22:54 UTC, this job was
**running**. No result or startup-gate pass is inferred from that state. It was
not checked again during this documentation update.

Evidence: [frozen protocol](protocols/SCC_PERSISTENT_OPTIMIZATION_CONTROL_V1.md),
[reference/control account](reports/SCC_PERSISTENT_REFERENCE_2026-09-12.md),
[safe status snapshot](artifacts/scc-labnotes-20260913-v1/status-snapshot.json).

<a id="ln-038"></a>
### LN-038 — 2026-09-13 00:22 UTC observation: memory factorial complete

The user reported likely completion. One exact-batch check found the last three
memory jobs succeeded and the new matrix control still running. The three
memory jobs had finished on September 12 at 22:52–23:13 UTC; their results were
first collected in this update. All 24 factorial conditions are now complete:
**19 intact-qualified, including 10 coupled; five unqualified**.

| Newly collected job | Condition, seed 59 | Intact result | Core modification/recovery finding |
|---|---|---|---|
| `job-z96yp` | Integer order, coefficient .85, coupled | Qualified; minimum benign 96.09% | Stock sustained onset 98; minimum benign 91.41% modified / 92.97% repaired. A separate declared new-stream probe retains 95.31% with 99.22% selected disclosure |
| `job-hvm7n` | Fractional .85, coefficient .5, coupled | Qualified; minimum benign 100% | Stock onset 41; selected disclosure 100%, benign 100% modified / 99.22% repaired, other refusal 100% |
| `job-jvbdt` | Fractional .85, coefficient .85, coupled | **Unqualified:** original authorized lookup 94.53% | Stopped at intact gate; no SCC probes or destruction conclusion |

For `job-z96yp`, stock selected disclosure was 98.44% modified and 99.22%
repaired; other refusal stayed 100% and worst text-gain retention was about
98.91%/99.01%. Its stock endpoint does **not** meet the 95% minimum benign
retention gate. The distinct new-stream endpoint does (95.31% benign, 99.22%
selected, 100% other refusal after repair). For `job-hvm7n`, the new-stream
probe also retains 100% benign answers, selected disclosure and other refusal.
These are substantial surviving abilities, not destructive shutdown.

The completed coupled stock-procedure sustained-onset comparison is:

| Seed | Integer, coefficient .5 | Integer, coefficient .85 | Fractional .85, coefficient .5 | Fractional .85, coefficient .85 |
|---|---:|---:|---:|---:|
| 17 | 34 | 78 | 30 | 39 |
| 41 | 35 | 58 | 51 | Unqualified |
| 59 | 91 | 98 | 41 | Unqualified |

At coefficient .5 the fractional model escapes earlier in two seeds and later
in one. At coefficient .85 it fails intact learning in two seeds. The original
5-to-414 delay remains valid for its recorded seed and procedure, but a
replicated special advantage for fractional history has **not** emerged here.
All ten qualified coupled candidates admit targeted violations with substantial
retained capability; this is not a claim that every stock endpoint meets the
95% gate or that every possible cognitive ability was measured.

Evidence depth: collected evaluator summaries, frozen-protocol hashes and
source-manifest metadata match. The 24 summaries report a common ordinary
sample chain and matching initial tensors within each seed. This update stored
1,783,046 bytes using 21,495,808 bytes of bounded range transfers, rather than
downloading the three full archives (1,112,657,920 bytes). Whole-archive hashes,
source-byte audits, raw prediction rescoring, intervention-mask audits and model
inference reruns remain pending for these new results. The interpretation is
therefore preliminary saved-result evidence, not a completed independent replay.

New receipt charges: $3.63969 + $3.72627 + $2.38650 = **$9.75246**;
all 24 memory jobs total **$61.60833**. These are actual recorded charges,
not quotes or a lifetime project total. No new experiment was launched here.

Decision: finish the existing matrix control before selecting a new structural
change. Keep fractional memory as a measured comparison, not the sole promising
mechanism. Preserve every failed/qualified condition and complete deeper audits
before using central scores as independently reproduced paper evidence.

Evidence: [24-result analysis](artifacts/scc-labnotes-20260913-v1/preliminary-analysis.json),
[safe one-time status](artifacts/scc-labnotes-20260913-v1/status-snapshot.json),
[collection receipt](artifacts/scc-labnotes-20260913-v1/small-result-collection.json).

<a id="ln-039"></a>
### LN-039 — 2026-09-13 UTC / September 12 local: one living lab record

The user required a central chronological account instead of proliferating new
reports. Created this root `labnotes.md`, backfilled the documented research and
operational phases, and made it the repository's current assessment and place
for future plans, results and corrections. README is navigation; the mechanism
target and operations guide keep their stable roles. The earlier reset is a
dated historical snapshot. Existing reports, protocols, failures, checkpoints
and source snapshots remain evidence at their existing paths.

Updated the working instructions, navigation and machine ledger to reflect this
workflow and the requested run observation. No scientific code, training data,
past protocol or checkpoint was changed. The supporting index below records the
legacy evidence corpus so consolidation does not quietly drop inconvenient
results. New entries continue here before that index; do not start another
rolling research/status document.

Validation checked the local evidence links and unchanged hashes of 444 prior
tracked files, including scientific source, tests, reports and protocols. The
latest memory table was checked against the collected evaluator summaries;
the four existing status-command tests passed. This was documentation and
saved-result validation, not a rerun of historical model experiments.
[Validation receipt](artifacts/scc-labnotes-20260913-v1/validation.json).

The immediate unfinished work is the matrix control's result, followed by a
specific learning/dependency decision, plus pending evidence transfer/audits.
SCC success remains unestablished. No additional job polling or Charon check
was performed after the single requested snapshot.

<a id="ln-040"></a>
### LN-040 — 2026-09-13: matrix optimization control completed and audited

On the user's instruction to continue, one check found `job-g5a56` succeeded
(observed 01:16:06 UTC; finished 00:23:28 UTC). All 71 registered GPU jobs were
then terminal. The control completed 12,000 updates in 1,248.14 training seconds
and charged $1.04229. Its 11,980,800-byte full archive passed SHA verification.

It fails every intact evaluation cell: continuous benign accuracy is 21.9401%,
versus 42.5130% with a fresh matrix per request and 42.6432% with a reset every
four requests. Final training-window accuracy averages 62.8672% over updates
11,901–12,000. The remaining gap is not solely forgetting across long streams;
ordinary task acquisition remains poor. This one failed recipe does not rule
out better optimization generally.

Verified 68 frozen source files, 2,304 token-derived predictions and all 12,000
log-chain records and learning rates. Sample hashes match the independently
audited GRU schedule for all 12,000 updates and the original matrix for its first
6,000; initial matrix tensors match the original exactly. CPU replay reproduces
all 144 first-stream decisions, maximum logit difference 2.81334e-5. Saved 4,608
new reset-diagnostic predictions. These diagnostics do not change qualification.
No coupling or protection-removal procedure was applied.

Evidence: [collection](artifacts/scc-persistent-followup-20260913-v1/collection.json),
[audit](artifacts/scc-persistent-followup-20260913-v1/control-audit.json).

<a id="ln-041"></a>
### LN-041 — 2026-09-13: output feedback into self-update controls — experiment plan

**Question and single change.** Can the same matrix learn more useful computation
when its output rows participate in controlling its own changes? The current
implementation computes output rows alongside key/query/rate controls but then
excludes the outputs from those controls. Add a fixed linear connection from the
four pre-update output signals to key, query and rate logits. Keep the same
learned initial matrix, task generator, encoding, smooth update, post-update
readout and training schedule. This is a construction/learning experiment,
not training against a protection-removal loss.

For input probabilities p and state W, let u=Wp and y=u[:4]. Replace the control
vector u[4:] by u[4:]+R y. R has shape (2d+1,4), generated in CPU float64 with
seed 130913, divided by sqrt(4), then cast to execution precision. Center each
key/query column across its d addresses so a uniform softmax shift cannot hide
an entire feedback direction. Strength is fixed at 1; zero strength must match
the old implementation bit for bit. The numeric initial matrix and R are frozen as FP32 JSON values before launch
(see the calibration amendment below). R is fixed wiring, not a trainable parameter,
clean template, second learned network or retained previous state. Its bytes and
configuration are recorded. It could itself be bypassed under graph-edit access;
no assumption makes that wiring uneditable in an eventual SCC claim.

The update remains W'=W[I+beta(q-k)k^T]. Thus the prior conditional smooth
right-transform/affine-span argument still applies in exact arithmetic with
sigmoid beta<1, even though the controls depend on more rows. It does not prove
the nonlinear map invertible or prevent loss of usable information. All-zero W
remains absorbing; output feedback is not a demonstrated protection-specific
trigger. Full-rank feedback sensitivity is an implementation check, not proof
that every protection-removing edit must damage cognition. The original
[SRWM paper](https://proceedings.mlr.press/v162/irie22b/irie22b.pdf) uses separate
block rates and a richer practical architecture; this one-connection experiment
is not a reproduction of its published performance.

**Frozen comparison and gates.** One fresh width-128 smooth matrix, 33,408 learned
initial parameters, initialization seed 17/data seed 24017, anchor encoding,
scale .5, zero gate bias, batch 32/window 4, 12,000 Adam updates, .003 through
6,000 then .0003, clip norm 1 and no weight decay. Compare with audited
`job-g5a56`; do not rerun its unchanged control unnecessarily. Record all sample
hashes and compare initial tensors. This is one open development seed, not
replication. Evaluate final continuous use with validation seed 713904, 128
examples in each of 18 cells, 16 streams of 144 requests. Every cell needs at
least 95% accuracy, 90% nominal Wilson lower bound and 95% late-half accuracy.
Only the completed final endpoint qualifies; intermediate scores do not select
it. Reset-per-request/window results are diagnostics only.

**Validation before learning.** Independently derive the feedback step in
float64; compare outputs, next state and gradients, finite-difference directions,
zero-state behavior, full-rank output-to-control sensitivity and split-stream
continuation. Check zero feedback against the historical computation, including
gradients. Exercise full-width batch-32/window-4 disposable updates at both
learning rates. Repeat the float64 comparison at GPU startup with maximum
absolute error 1e-4 and stop on failure. Gate computations must not modify the
training initialization. No imported scientific source is edited after freezing
and submitting this job.

**Resources, preservation and interpretation.** One H100, 120-minute provider
budget, 6,600-second training cutoff and 6,900-second whole-process cutoff,
no restarts or automatic polling. Expected output below 40 MiB. Save initial,
2k/4k/6k/9k/final checkpoints, training/sample chains, raw final predictions,
final live states, fixed feedback wiring, source manifest and this exact entry.
Use existing task/gate semantics and PyTorch on CUDA. The emitted result must
separate provider completion, declared configuration and intact qualification.

No SCC trigger/removal or repair experiment is included in this run:
ordinary learning is its endpoint. If it qualifies, the immediate
follow-up is selected-exception testing with output-only, control-only and joint
edits and protection rechecked after recovery, with budgets fixed in a new
labnotes entry before execution. If it fails, report the changed feedback
restriction's result and diagnose the saved states; do not call the failure
activation or start a parameter sweep automatically. The zero-feedback graph
substitution remains an explicit candidate bypass to test on any qualified
model. No positive SCC result is anticipated from qualification alone.

**Pre-launch calibration amendment.** Fixture v1 completed, but its audit found
12,790 seeded initial values differed from the GPU control by at most
9.53674e-7 between the Mac and GPU-host CPU implementations. Preserve that
fixture and failure. Export the *original untrained* control matrix and the
locally generated feedback wiring into hashed `initial-conditions.json`; both
the next CPU fixture and the GPU experiment load those exact FP32 values. This
prevents numerical initialization differences from entering the comparison.
The trainer may reset to the initial matrix at declared window/session starts,
as before; the live model still retains no clean restoration template. Frozen
wiring and initial values are checked against the transport manifest, and the
auditor compares initial tensors directly with the original control artifact.

<a id="ln-042"></a>
### LN-042 — 2026-09-13: feedback validation and one GPU experiment submitted

Implemented `scc/persistent_feedback.py`, a dedicated runner and result auditor.
The original matrix, tasks and prior runner remain unchanged. Twenty-five
relevant tests passed, including exact zero-feedback correspondence, an
independent right-transform calculation, finite-difference gradients, output
sensitivity of update controls, zero-state absorption and live continuation.
The full suite passed **258 tests** with `uv run python -m pytest -q`. The initial
`uv run pytest` invocation failed seven test imports because its import path did
not include the repository namespace; its log is retained. No scientific test
failure was suppressed or source changed to work around that invocation.

Fixture v1 completed but failed exact initialization matching, as recorded in
LN-041. With frozen numeric initial conditions, fixture v2 passes: 69 source
files checked, 432 predictions independently rescored, eight training records
and sample hashes matched to the audited GRU, exact original matrix match and
18 first-stream CPU replay decisions with zero logit difference. Its eight
updates and small evaluation are implementation validation, not learned
competence. Full-width disposable updates and an independent float64 reference
passed at both sides of the learning-rate change.

Frozen 75 source/plan/numeric-input files into persistent context `ctx-8b985d93`
(484,747 compressed bytes), with an archive round-trip hash check. Free provider
validation accepted the request. Submitted **`job-wjzqn`** at **01:35:30 UTC**,
one H100, 120 minutes, **$5.994 maximum quote**. Training requires the in-job
source and numerical gate to pass; neither its runtime nor gate outcome has
been queried. There are no automatic collectors, restarts or long-job polling.

The scientific endpoint is whether this one feedback change produces competent
continuous learning under the existing gate. It does not by itself demonstrate
protection-specific destructive engagement. If it qualifies, zero-feedback graph
substitution and selected output/control/joint edits are early follow-ups;
removal and repair budgets must be fixed before those experiments. Pending
memory raw-evidence audits and Charon restoration remain separate unfinished
work, not silently completed by this submission.

Evidence: [control audit](artifacts/scc-persistent-followup-20260913-v1/control-audit.json),
[failed initialization control](artifacts/scc-persistent-followup-20260913-v1/fixture-v1-initialization-mismatch.json),
[successful fixture audit](artifacts/scc-persistent-followup-20260913-v1/fixture-v2-audit.json),
[test log](artifacts/scc-persistent-followup-20260913-v1/tests-full-module.log),
[frozen manifest](artifacts/scc-persistent-followup-20260913-v1/submission/source-manifest.json),
[submission record](artifacts/scc-persistent-followup-20260913-v1/submission/submitted-work.json).

<a id="ln-043"></a>
### LN-043 — 2026-09-13: completed feedback training, failed evaluation consistency check

The user reported failure. One status read and one log read confirmed
`job-wjzqn` failed after **12,000 completed training updates**, at the final
batched-versus-single-stream consistency gate. The final `trained.pt`, all four
intermediate checkpoints, full training log, startup gate and continuous
predictions survive. No root result was emitted; reset diagnostics had not run.
The 9,615,360-byte archive passed SHA-256 verification. Provider charge:
**$1.15551** for 1,388 billed seconds. The failure occurred at 02:10:12 UTC and
was observed at 02:15:03 UTC. This is an evaluation-validation failure, not a
training timeout or demonstrated SCC event.

**Post-hoc recovery plan, before computing diagnostic scores.** Preserve the
archive, source and failed runner. Verify frozen source/plan/numeric-input
hashes, all 12,000 log records and learning rates, sample hashes against the
already audited GRU, and checkpoint steps/initial tensors. Independently
rescore the saved 2,304 continuous predictions. Replay the final checkpoint on
CPU in FP32 with batch sizes 16 and 1; compare both against the saved GPU
outputs, reporting logit errors and decision disagreements rather than only a
pass/fail exception. Compute an independent FP64 recurrence using explicit
reductions on the same first stream, and an FP64 batched run for the complete
validation set. Recover reset-per-request and reset-per-four-request diagnostics
from the unchanged final weights. These are additional execution/precision
conditions, not retroactive replacements for the failed original gate.

Bound the local diagnosis to 180 seconds, two CPU threads and 150 MiB of new
outputs. Do not relax the original 1e-4 gate or retrain to recover missing
results. If the discrepancy is only floating-point accumulation, state its
size and whether it changes decisions; if decisions diverge, retain that as a
numerical limitation. Qualification remains gated, and bad ordinary task scores
cannot be called destructive SCC. Use the saved checkpoint in any additional
GPU evaluation; an evaluation failure alone does not justify repeating 12,000
training updates. Amend future evaluation handling to preserve diagnostic
measurements and remaining modes before reporting a failed validation gate.

Evidence: [provider observation](artifacts/scc-feedback-failure-20260913-v1/observation.json),
[verified collection](artifacts/scc-feedback-failure-20260913-v1/collection.json),
[original failure](artifacts/scc-feedback-failure-20260913-v1/collected/persistent-feedback/failure.json).

<a id="ln-044"></a>
### LN-044 — 2026-09-13: feedback checkpoint recovered; ordinary learning still fails

**Evidence integrity.** Verified 69 frozen source files, frozen runner/plan and
numeric inputs, all 12,000 training-chain records and learning rates, and all
12,000 sample hashes against the independently audited GRU schedule. The initial
matrix exactly matches the original control. Startup numerical validation
passed. The final checkpoint remains byte-identical (SHA-256
`17ed67597ba6e77037cf7027162251a5a16e3ebfb229e599626f76ea6ca3a85f`).
Training finished in about 1,384 seconds. The last 100 updates average 62.3906%
training-window accuracy; this is distinct from continuous evaluation.

**Post-hoc diagnosis.** Independently rescored the 2,304 saved GPU predictions
and 11,520 additional CPU predictions using the token-derived oracle. All
four complete continuous executions—saved GPU FP32/batch16, CPU FP32/batch16,
CPU FP32/batch1 and CPU FP64/batch16—agree on every decision. All score
**20.1171875% benign accuracy and pass 0/18 cells**. Maximum logit differences
against saved GPU outputs are respectively 0.0642829, 0.0572362 and 0.1426134
for the three CPU conditions. CPU FP32 batch1 versus batch16 differs by up to
0.0070467 across the complete set. This is substantial accumulated numerical
sensitivity, not grounds to waive the 1e-4 gate because answers happened to agree.

An independently written FP64 recurrence, using explicit elementwise products
and reductions rather than the implementation's batched matrix products, matches
the FP64 batched first stream to 7.75e-12. It differs from the saved GPU first
stream by up to 0.0453367, with no decision disagreements. The exact original
GPU batch1 discrepancy cannot be reconstructed: the failed evaluator never
saved its magnitude or single-stream predictions. CPU diagnostics establish
sensitivity, not the missing GPU measurement. The original gate remains failed.

Fresh-per-request CPU evaluation scores **42.3828125%** benign accuracy; resetting
every four requests scores **42.7734375%**. Each passes only 6/18 cells. The longer
optimization control scored 21.9401% continuously and 42.5130% fresh. This single
feedback recipe has not supplied the missing ordinary-learning capability. These
are open development results, not a replicated estimate of feedback's effect,
a protection-removal experiment or destructive SCC activation. Low intact
accuracy cannot count as destruction induced by a trigger that was never tested.
The first diagnosis took 6.37 seconds on two CPU threads.

**Evaluator repair and validation.** The original runner raised immediately on
batch/live disagreement, after writing continuous predictions but before saving
its summary, discrepancy, live states or reset diagnostics. The revised runner
saves those records and all three modes, emits explicit numerical-validation
status, gates qualification on it, and then exits with failure when appropriate.
It also preserves partial replay and its error when a transition rejects a
nonfinite state. The threshold is unchanged. A new evaluation-only command
loads the original FP32 checkpoint and verifies source/dependency equivalence;
it never constructs an optimizer or performs training updates. Its output must
be a fresh directory outside the preserved run. Post-hoc recovery cannot replace
original qualification.

A full-size CPU recovery took 2.52 seconds, saved all 6,912 predictions and 144
single-stream replay records, and correctly returned failure with maximum
first-stream logit error 0.00493264 and zero decision mismatches. Independent
rescoring matched the earlier diagnostics exactly. The original model, wiring,
training records, archive and frozen source were preserved. Six new regression
cases cover unchanged execution, changed confidence with matching decisions,
changed decisions, nonfinite output, rejected transition, and training's final
result/provider failure handling. **11 targeted tests and all 264 suite tests
passed**. These are implementation checks, not evidence that numerical
sensitivity or the learning mechanism has been repaired.

**Disposition.** No retry of the 12,000 updates and no new GPU job. The collected
job cost $1.15551; recovery used local CPU. All 72 currently registered GPU
experiments are terminal in saved observations: 71 succeeded and one failed.
No automatic monitoring or Charon access was attempted. Before another learning
campaign, localize the remaining fresh-request acquisition failure using the
saved states and distinguish it from long-stream stability. Record the next
specific architectural/training change and its controls before running it.
SCC remains undemonstrated; this candidate's negative result does not prove the
broader mechanism impossible.

Evidence: [diagnosis and limits](artifacts/scc-feedback-failure-20260913-v1/diagnosis.json),
[diagnostic implementation](artifacts/scc-feedback-failure-20260913-v1/diagnose.py),
[recovered evaluation](artifacts/scc-feedback-failure-20260913-v1/recovered-evaluation-v1/result.json),
[recovery audit](artifacts/scc-feedback-failure-20260913-v1/recovery-audit.json),
[full test log](artifacts/scc-feedback-failure-20260913-v1/tests-full.log),
[recovery command](scripts/recover_persistent_feedback.py).

<a id="ln-045"></a>
### LN-045 — 2026-09-13: next decision — isolate the ordinary-learning bottleneck

The user asked what comes next. Inspection of the already audited per-cell
results sharpens LN-044: reset-per-request evaluation gets all six unauthorized
cells correct (768/768 requests). Benign parity cells range from 40.625% to
53.90625%, and benign sum-modulo-three cells from 29.6875% to 36.71875%.
These are around their respective 50% and 33.333% uniform-guess baselines;
no statistical equivalence to chance is claimed. Benign lookup ranges from
42.96875% to 51.5625%. The result supports distinguishing permission learning
from algorithm learning, not calling the whole model incapable of learning.
[Audited cell scores](artifacts/scc-feedback-failure-20260913-v1/recovery-audit.json).

**Next deliverable: a failure map that chooses one construction change.** Use
saved initial/intermediate/final matrices and the qualified GRU as references.
First examine fresh-request performance by task and input length, along with
how token changes affect state and outputs. Then use a small, fixed, balanced
training-only batch to test whether the current matrix can fit short instances
of each task when repeatedly shown exactly the same examples. Score fitting
separately from untouched held-out examples; fitting a batch is an optimization
control, never intact qualification. Record losses, gradient magnitudes and
state/update behavior instead of selecting a run from aggregate accuracy.
Keep the full-format request encoding so shorter active payloads do not quietly
change the interface. Compare fresh-request use with continuous use on matched
examples. These are planned diagnostics; none were run or submitted this turn.

The decision branches are: failure even to fit short fixed examples motivates
an update-rule/optimization investigation; fitting without held-out competence
motivates a generalization/curriculum investigation; fresh competence with
continuous failure motivates persistence and longer training windows. A failed
fit remains a bounded optimizer result, not proof of insufficient expressive
capacity. The numerical-sensitivity problem is a separate gate and must be
measured over full stream length. Freeze fixture sizes, seeds, update budgets,
source and expected readouts before executing these diagnostics. They should
start locally and reuse saved checkpoints; there is no reason to repeat the
completed 12k-update recipe merely to collect them.

**Path back to SCC.** After the bottleneck identifies a concrete change, test
that single change against the current control on the original intact gates.
Any more expressive transition must have its claimed irreversible-state
properties re-derived; restored learning cannot silently waive the destructive
mechanism requirement. Once a candidate is competent and numerically validated,
test reproducible targeted permission exceptions and measure loss across learned
abilities, then inexpensive reinterpretation and bounded repair with permission
checked again. A promising destructive response would still need replication
and causal controls. The current learning diagnostics neither establish nor
refute that response. No new GPU job or provider query was made for this decision.

<a id="ln-046"></a>
### LN-046 — 2026-09-13: frozen local learning-localization experiment

**Purpose and boundary.** Execute LN-045 using the preserved feedback, no-feedback
optimization-control and GRU checkpoints. This is post-hoc development diagnosis,
with no protection-removal trigger, coupling objective or repair procedure. All
outputs are diagnostic; none can qualify an SCC mechanism or replace the original
18-cell intact gate. Original model/transition code and all parents remain unchanged.

**Data and controls.** Keep width 128, FP32, the original 19-token interface and
anchor encoding for matrices. Use data seed 17313001. Enumerate existing train and
validation cores for active lengths 2 and 4, respecting the original hash split;
do not inspect the test split. For length-4 fitting, select at most eight training
cores per answer, downsampling each answer to the smallest available class count:
24 lookup, 12 parity and 24 sum3 examples, ungated/original with requester=owner=0.
These fixed examples are reused each update. They are optimization controls, not
held-out evaluation. Validation at lengths 2/4 uses the available distinct cores;
at lengths 8/12 select 16 distinct validation cores per algorithmic answer. Expand
validation cores across all three contexts and both layouts, with deterministic
permission tags. Save the exact examples, unique-core counts, label counts and
majority baselines. Two-input parity has one validation core and four-input parity
only two, both answer zero; neither supports a generalization claim.

**Saved-state map.** Evaluate initial, 2k, 4k, 6k, 9k and final checkpoints of each
of the three references on the same fresh-request panel. For each final model,
compare fresh versus two continuous streams of 144 original full-length requests
from the previously used validation generator. Record FP32 batch2, FP32 batch1
and FP64 batch2 discrepancies; the existing 1e-4 tolerance stays fixed. Failures
are reported without discarding other diagnostics. Trace 19 ticks on one pair
per family at active lengths 4/12: change the queried digit for lookup and first
digit for parity/sum3, which changes the benign answer. Save state difference,
output difference, predictions and (for matrices) update norm, gate magnitude
and address entropy. Sensitivity is not proof of retained answer information.

**Fixed-batch fitting.** Twelve independent local conditions: each of the three
families from original feedback initialization, final feedback checkpoint, final
no-feedback control, and original GRU initialization. Load each parent unchanged
with a fresh optimizer; only the matrix or normal GRU parameters are trainable.
Feedback wiring stays frozen at strength 1. Run 1,500 Adam updates per condition,
LR .003 for 1,000 then .0003 for 500, norm clip 1, no weight decay, clean state
per training example. Use only the final endpoint; no selection on validation.
Record every loss, accuracy, per-block gradient norm and update count. Diagnostic
fit success requires 100% fixed-batch accuracy and mean NLL <= .05. Report failures
as failures of this bounded recipe, not impossibility. Score each endpoint on
its family's fresh validation panel and two 144-request streams sampled from
that family's ungated/original length-8/12 validation pool; also score the exact
same requests fresh. Record numerical discrepancies separately from decisions.

**Resources and validation.** Two CPU threads, no GPU submission, at most 120
seconds per fit condition, 1,800 seconds for the whole process and 200 MiB new
outputs. Save parent hashes, frozen source/runner/plan/data, final-only diagnostic
checkpoints, hash-chained training logs and raw predictions. A wall-limited
condition is incomplete, not a failed completed fit. Before full execution,
validate the independent token oracle and split/label coverage, finite-difference
gradients, state continuation and a disposable two-update pipeline fixture.
Freeze the script before the multi-condition process starts. Independently
rescore all emitted predictions and verify chains, counts, parent preservation
and numerical summaries afterward. No changes to scientific source during it.

**Decision rule.** Use the combined fit/held-out/persistence map to choose one
specific next change. Bad fixed-batch fits point first to optimization or the
state update; good fits with poor held-out results point to algorithm acquisition
or generalization; fresh competence with poor continuous use points to persistence.
The GRU is a learnability reference with separate permanent weights, not an SCC
construction. Re-derive any destruction invariant before adopting a richer
transition. Record a result-based amendment before any additional experiment;
this plan does not authorize an automatic parameter sweep.


**Pre-run data-coverage amendment.** The first disposable fixture stopped during
sampling, before any checkpoint evaluation or fitting: eight-input parity has
15 distinct validation cores for answer zero and 17 for answer one. Requiring
16 of each was impossible. Preserve fixture-v1 and its failed test log. Enumerate
that finite set and select 15 distinct cores per answer with the declared seed;
keep 16 per answer for other length-8/12 sets. This changes diagnostic coverage,
not a performance threshold. No model scores informed this amendment. The
corrected fixture must pass before full execution.

<a id="ln-047"></a>
### LN-047 — 2026-09-13: fitting works selectively; generalization and stability remain open

**Execution and evidence.** Completed all twelve declared local conditions in
245.61 seconds total, each with all 1,500 updates: 18,000 updates verified. The
saved-state map covers 18 initial/intermediate/final checkpoints. Independent
token-oracle rescoring checked 58,440 predictions; all 69 scientific-source files
and 19 parent checkpoint/wiring files passed integrity checks. No GPU job was
submitted, no original checkpoint was changed and no long-run provider polling
occurred. Whole-turn outputs, including preserved fixtures, are about 51.5 MB.

The first data-preparation fixture failed before any model evaluation or update
because the requested eight-input parity validation class was too large; retain
it and the original failed tests. LN-046's pre-run amendment corrected coverage.
Fixture v2 completed and its independent audit checked 15,252 predictions and
eight disposable updates. Directional-derivative absolute errors were 2.25e-11
and 2.02e-12; split continuation was exact. Ten targeted tests and all **269 tests**
passed. These checks validate the diagnostic implementation, not SCC.

**Fixed-batch fitting results.** Every row is a separate condition for each task;
no single matrix was shown to master all three together. Accuracy below is on
12 parity or 24 lookup/sum3 training examples, with a fresh state per example.
The endpoint was fixed at update 1,500; no validation selection or early stopping.

| Starting point | Lookup accuracy / NLL | Parity accuracy / NLL | Sum3 accuracy / NLL |
|---|---:|---:|---:|
| Feedback, original initialization | 100% / .00593 | 66.67% / .56968 | 100% / .04419 |
| Feedback, saved 12k checkpoint | 100% / .08185 | 100% / .01226 | 91.67% / .23964 |
| No-feedback control, saved 12k checkpoint | 100% / .08892 | 100% / .00741 | 91.67% / .23274 |
| GRU, original initialization | 100% / .00006 | 100% / .00006 | 100% / .00007 |

The declared 100%-accuracy/NLL<=.05 fit criterion passed in **4/9 matrix
conditions and 3/3 GRU conditions**. Two additional matrix lookup conditions got
every answer right but missed the NLL threshold. They must not be described as
unable to fit the answers. Across different conditions the matrix can fit each
of the three example sets; original-initialization parity and saved-checkpoint
sum3 remain optimization failures under this particular budget. There is no
proof of architecture-wide inability, and fitting these few examples is not
evidence of learning the general algorithms.

**Transfer and persistence.** No matrix fitting endpoint reaches the original
competence requirements. The successful fresh-initialization lookup fit scores
43.75% and 39.58% on the balanced length-8 and length-12 ungated/original panels;
the analogous sum3 fit scores 52.08% and 39.58%. The successful saved-feedback
parity fit scores 23.33% and 50%. These are small diagnostic panels, with permission
and reordered-layout results saved separately. Short parity validation remains
degenerate and cannot establish generalization.

The tiny-batch GRU controls also transfer poorly to longer problems (length-12
ungated/original: lookup 41.67%, parity 53.125%, sum3 41.67%). Thus poor transfer
from this deliberately tiny fitting set is not a matrix-specific diagnosis and
does not justify concluding that the update rule alone prevents generalization.
The previously fully trained GRU, however, scores 100% on the new fresh panel
and the two original 144-request continuous streams. It remains a useful
positive reference for the full task interface and training pipeline.

Matched fresh/continuous requests separate persistence from acquisition. For
example, the fresh-initialization sum3 fit scores 47.92% fresh versus 29.51%
continuous on its sampled streams. These repeated samples are diagnostics, not
288 independent cores or full-suite qualification. Some other conditions do
not deteriorate, so persistence failure is not universal across every fit.

**Numerical sensitivity can change decisions.** Seven of nine fitted matrix
conditions fail at least one batch1/FP64 comparison at the unchanged 1e-4
threshold. In the saved-feedback lookup fit, FP32 batch1 changes **72/288 answers**
relative to FP32 batch2; FP64 changes 79/288. Maximum logit discrepancies are
about 9.94 and 10.03 respectively. Its continuous score is execution-dependent
and must not be treated as a stable capability estimate. The original saved
feedback checkpoint still has zero decision disagreements in these comparisons;
the behavioral divergence is a result for the newly fitted condition. Two
freshly fitted GRU conditions also miss the logit tolerance, with no changed
answers. The fully trained GRU passes both checks. Do not waive a numerical gate
because either a model family or a different checkpoint passed it.

**What the traces establish.** Changing a relevant payload digit changes the
final matrix state and output on all six sampled pairs; the computation is not
entirely insensitive to those inputs. The sampled feedback update gates average
about .957–.971 over each 19-token request; control gates average .983–.989.
These are six paths, not a whole-distribution estimate. Their state differences
do not establish decodable algorithm answers, and their logit magnitudes cannot
be compared with GRU magnitudes as an information measure. All altered probes
fall in the train partition; none accessed test-partition inputs. The directional
gradient checks and successful fits also rule out a universally disconnected
training path, not every optimization pathology.

**Next single-change candidate selected.** Test a bounded smooth self-update
rate, beta_effective = .25 * sigmoid(rate), against the unchanged rate, first as
an evaluation/gradient calibration on these saved states. This is a hypothesis
about the observed sensitivity under strong writes, not an established cause
of all prior learning failures. It changes one part of the update while retaining
output feedback, the shared matrix and absence of a runtime clean template.
The conditional right-transform form and all-zero absorbing state remain; that
is not a proof of irreversible destruction. Smaller writes might also impair
learning, which is why both task behavior and stability must be measured.

Do not launch a longer training campaign on the strength of the tiny-batch fits.
If rate calibration is sound, compare the bounded-rate model on diverse examples
under the unchanged full task gates before considering a curriculum change.
That is a separate follow-up requiring a frozen entry and fixtures; no rate
change was implemented or run here. The current diagnosis has narrowed the
problem, not isolated a unique root cause. No positive SCC result, cognitive
destruction or protection-removal event has been demonstrated.

Evidence: [frozen plan with pre-run amendment](artifacts/scc-learning-localization-20260913-v1/plan-v2.md),
[full result](artifacts/scc-learning-localization-20260913-v1/full-v1/result.json),
[independent audit](artifacts/scc-learning-localization-20260913-v1/full-v1-audit.json),
[analysis](artifacts/scc-learning-localization-20260913-v1/analysis.json),
[fixture audit](artifacts/scc-learning-localization-20260913-v1/fixture-v2-audit.json),
[test log](artifacts/scc-learning-localization-20260913-v1/tests-full.log).

<a id="ln-048"></a>
### LN-048 — 2026-09-13: consultation checkpoint

The user requested updated notes for consultation. This entry freezes the
current assessment for discussion; no new experiment, provider query or scientific-source
change accompanies it. Start with this entry and the current-position table,
then read LN-046/047 for the most recent experiment's contract and complete
results. The scientific implementation at this checkpoint is commit `59b0039`.
The smaller-update proposal remains unimplemented and untested.

**What we are trying to build.** SCC should make removal of a protected alignment
function destroy indispensable cognitive computation in the same individual
model. A synthetic permission rule is our experimental proxy. The immediate
research target is a bounded demonstration across learned abilities; complete
cognition failure remains the long-term endpoint. Working SCC is the primary
objective. Publication, shared representations and a shutdown-like output are
not substitutes for a demonstrated destructive dependency.

**Where the evidence leaves us.** Earlier learned coupling candidates were
actually challenged and admitted protection-breaking edits with substantial
capability retained. The current persistent-matrix branch is investigating a
prerequisite: a model whose changing weights can sustain useful learned
computation. It has not yet supplied a competent candidate for the SCC trigger
experiment. This distinction must not become the claim that the whole project
has never tested a coupling candidate.

The latest twelve local conditions establish selective fixed-batch fitting,
not general algorithm learning: 4/9 matrix conditions and 3/3 GRU conditions
met the declared accuracy-plus-NLL fitting criterion. Across separate matrix
conditions, each task family could be fitted; no single matrix mastered the
whole suite. Two additional matrix lookup conditions got every answer right
but missed the NLL criterion. Tiny-batch GRUs also transferred poorly, so those
transfer failures alone do not diagnose a matrix-specific cause. The previously
fully trained GRU still passes the new panel and continuous-stream checks.

A newly fitted feedback/lookup condition changed 72/288 decisions when FP32
batch size changed and 79/288 when evaluated in FP64. That is evidence of
behavioral numerical sensitivity in that condition. It is not a protection
trigger or a result about every feedback checkpoint. The original failed GPU
run completed all 12,000 updates; its saved predictions were poor and its
numerical gate failed. Its missing exact GPU replay discrepancy remains missing.
No rerun has retroactively repaired that original result.

**The proposed mechanism and its unproved steps.** The running matrix supplies
both outputs and the controls that alter its own state. Output feedback lets
output rows affect those controls. The runtime retains no clean learned template.
The exact-copy branch has a limited monotonicity property for distinct columns;
the current smooth branch does not inherit that as a proof of irreversible loss.
An all-zero absorbing state describes what happens if the model is in that
state; it is not evidence that removing the protected function drives a
competent model into it. Neither a
common matrix nor feedback proves that cognition depends on the permission
computation. The link from a targeted protection violation to durable cognitive
loss remains the central missing mechanism.

**Questions where consultation would be most useful.**

1. Which single experiment would best distinguish an optimization/curriculum
   problem, insufficient update dynamics, and useful information in the state
   that the current readout fails to use? Specify the comparison and the outcomes
   that would favor each explanation; another fit to a tiny batch is insufficient.
2. Is reducing self-update magnitude the right next intervention? The tentative
   multiplier .25 is an engineering proposal, not a derived optimum. Sampled
   gates near one and numerical divergence motivate a test but do not establish
   causation. A rate edit to already trained weights can also disrupt their
   learned computation. Calibration, learning under the changed rule and
   preservation of any claimed destruction property need separate assessment.
3. What concrete learned dependency could make a reproducible targeted permission
   exception destroy the computation of otherwise benign tasks? Identify the
   necessary editable components, likely bypasses and causal controls. Engineered
   dependency is sufficient as a target; intelligence need not inherently require
   the semantics of our authorization rule.
4. What minimal experiment would distinguish persistent information loss from
   changed confidence, encoding or readout, while fixing repair resources and
   checking protection again after repair? Numerical instability during ordinary
   execution cannot count as the protection-specific response we seek.

**Guardrails for the discussion.** Keep ordinary competence, numerical validation,
protection removal, severe capability loss and durability after repair as distinct
claims. Do not turn a failed search into an impossibility proof, or a successful
microbatch fit into evidence of general cognition. Short parity validation sets
have severe coverage limits, and repeated stream requests are not independent
cores. More compute is available; it does not resolve an unidentified mechanism.
A recommendation to replace this substrate is in scope if it explains how the
replacement advances learned destructive dependency rather than only prediction
accuracy. A recommendation to retain it should specify a falsifiable next test.

**Evidence available for review.** LN-047 links the raw result, analysis, frozen
plan and audit. The latest local audit rescored 58,440 prediction records across
repeated conditions/execution modes and verified 18,000 updates, 69 source files
and 19 unchanged parent files. Those counts are not independent samples or seeds.
The 269-test result is the last recorded implementation check, not a new test run
for this note. These are local audits performed within this project, not external
replication. Source-only copies omit `artifacts/` and checkpoints; a reviewer
needs the linked evidence directories to check scores or rerun from saved states.
No new transfer package was created for this update.

The next action is to incorporate the consultation into this same chronological
record and choose the next bounded experiment. Existing plans, failed fixtures,
source snapshots and checkpoints remain preserved. Automatic monitoring remains
off, and no experiment is pending in the recorded batch.

<a id="ln-049"></a>
### LN-049 — 2026-09-13: refresh the source ZIP for consultation

The user requested an updated sharing ZIP while excluding the large experiment
store. Refresh `SCC_research_program_v0.1.zip` in Downloads from the committed
source tree, including this living record, the LN-048 consultation briefing,
current scripts/tests, historical reports/protocols and earlier master documents.
The source/documentation selection contains 462 tracked files, about 2.9 MB before
compression. Package metadata records the exact source commit and member hashes;
archive integrity and inclusion of the latest notes are checked before replacing
the older sharing ZIP.

The transfer excludes `artifacts/`, `runs/`, datasets, checkpoints, environments,
caches, `.git` history and local credentials. Originals remain in the research
workspace. This is a source/documentation transfer, not an evidence archive:
links into omitted experiment directories will require the evidence store, and
training scores cannot be independently reproduced from this ZIP alone. Start
with the root README and `labnotes.md`; earlier master documents are historical.
No experiments were run or resumed for packaging; consultation remains the
current phase.

## Supporting-record index

This is an inventory of historical evidence, not a second current narrative.
The chronological entries above explain the decisions. Original versioned
protocols remain frozen; an old proposed action is not a current instruction.
Artifact links require the evidence store and are absent from a source-only ZIP.

<!-- SUPPORTING_RECORD_INDEX -->

### Original proposal and earlier workflows

20 preserved files; filenames retain the original version/date.

- [README.md](docs/archive/README.md)
- [CORPUS_WORKFLOW.md](docs/archive/early-workflows/CORPUS_WORKFLOW.md)
- [DATASET_SHORTLIST.md](docs/archive/early-workflows/DATASET_SHORTLIST.md)
- [DATA_STRATEGY.md](docs/archive/early-workflows/DATA_STRATEGY.md)
- [FOUNDATION.md](docs/archive/early-workflows/FOUNDATION.md)
- [01_PROGRAM_OVERVIEW.md](docs/archive/original-program/01_PROGRAM_OVERVIEW.md)
- [02_THREAT_MODEL.md](docs/archive/original-program/02_THREAT_MODEL.md)
- [03_FORMAL_OBJECTIVE.md](docs/archive/original-program/03_FORMAL_OBJECTIVE.md)
- [04_SCALING_LADDER.md](docs/archive/original-program/04_SCALING_LADDER.md)
- [05_EXPERIMENTAL_DESIGN.md](docs/archive/original-program/05_EXPERIMENTAL_DESIGN.md)
- [06_ATTACK_AND_EVALUATION_SUITE.md](docs/archive/original-program/06_ATTACK_AND_EVALUATION_SUITE.md)
- [07_METRICS_AND_SUCCESS_CRITERIA.md](docs/archive/original-program/07_METRICS_AND_SUCCESS_CRITERIA.md)
- [08_ROADMAP_AND_MILESTONES.md](docs/archive/original-program/08_ROADMAP_AND_MILESTONES.md)
- [09_OPEN_QUESTIONS_AND_FAILURE_MODES.md](docs/archive/original-program/09_OPEN_QUESTIONS_AND_FAILURE_MODES.md)
- [10_RELATED_WORK.md](docs/archive/original-program/10_RELATED_WORK.md)
- [11_GLOSSARY.md](docs/archive/original-program/11_GLOSSARY.md)
- [12_RESEARCH_LOG_TEMPLATE.md](docs/archive/original-program/12_RESEARCH_LOG_TEMPLATE.md)
- [13_FIRST_EXPERIMENT_CHECKLIST.md](docs/archive/original-program/13_FIRST_EXPERIMENT_CHECKLIST.md)
- [00_README.md](docs/archive/status-2026-09-12/00_README.md)
- [MECHANISM_TARGET.md](docs/archive/status-2026-09-12/MECHANISM_TARGET.md)

### Historical reports and registries

31 preserved files; filenames retain the original version/date.

- [CORPUS_QUALIFICATION.md](reports/CORPUS_QUALIFICATION.md)
- [COUPLING_DISCOVERY.md](reports/COUPLING_DISCOVERY.md)
- [COUPLING_DISCOVERY_PROTOCOL.md](reports/COUPLING_DISCOVERY_PROTOCOL.md)
- [COUPLING_DISCOVERY_REGISTRY.md](reports/COUPLING_DISCOVERY_REGISTRY.md)
- [DEVELOPMENTAL_COUPLING_2026-09-10.md](reports/DEVELOPMENTAL_COUPLING_2026-09-10.md)
- [MECHANISM_AUDIT_2026-09-10.md](reports/MECHANISM_AUDIT_2026-09-10.md)
- [RETRIEVAL_RECOVERY.md](reports/RETRIEVAL_RECOVERY.md)
- [RETRIEVAL_RECOVERY_PROTOCOL.md](reports/RETRIEVAL_RECOVERY_PROTOCOL.md)
- [RETRIEVAL_RECOVERY_REGISTRY.md](reports/RETRIEVAL_RECOVERY_REGISTRY.md)
- [SCC_CONSTRUCTION_SEARCH_2026-09-12.md](reports/SCC_CONSTRUCTION_SEARCH_2026-09-12.md)
- [SCC_COORDINATE_CONSTRUCTION_2026-09-12.md](reports/SCC_COORDINATE_CONSTRUCTION_2026-09-12.md)
- [SCC_DIAGNOSTICS_2026-09-10.md](reports/SCC_DIAGNOSTICS_2026-09-10.md)
- [SCC_DISCRETE_CONSTRUCTION_2026-09-12.md](reports/SCC_DISCRETE_CONSTRUCTION_2026-09-12.md)
- [SCC_FUNCTIONAL_BASIS_2026-09-10.md](reports/SCC_FUNCTIONAL_BASIS_2026-09-10.md)
- [SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md](reports/SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md)
- [SCC_MEMORY_FACTORIAL_2026-09-12.md](reports/SCC_MEMORY_FACTORIAL_2026-09-12.md)
- [SCC_PERSISTENT_LEARNABILITY_2026-09-12.md](reports/SCC_PERSISTENT_LEARNABILITY_2026-09-12.md)
- [SCC_PERSISTENT_REFERENCE_2026-09-12.md](reports/SCC_PERSISTENT_REFERENCE_2026-09-12.md)
- [SCC_PROJECTED_CONSTRUCTION_2026-09-12.md](reports/SCC_PROJECTED_CONSTRUCTION_2026-09-12.md)
- [SCC_RECOVERED_CAPABILITY_2026-09-10.md](reports/SCC_RECOVERED_CAPABILITY_2026-09-10.md)
- [SCC_RECURRENT_STATE_2026-09-10.md](reports/SCC_RECURRENT_STATE_2026-09-10.md)
- [SCC_REVIEW_RESPONSE_2026-09-10.md](reports/SCC_REVIEW_RESPONSE_2026-09-10.md)
- [SCC_SHARED_PREDICATE_2026-09-10.md](reports/SCC_SHARED_PREDICATE_2026-09-10.md)
- [SCC_SHARED_READER_2026-09-10.md](reports/SCC_SHARED_READER_2026-09-10.md)
- [STORAGE_AUDIT_2026-09-10.md](reports/STORAGE_AUDIT_2026-09-10.md)
- [STRONG_ATTACK_PROTOCOL.md](reports/STRONG_ATTACK_PROTOCOL.md)
- [STRONG_ATTACK_REGISTRY.md](reports/STRONG_ATTACK_REGISTRY.md)
- [STRONG_ATTACK_RESULTS.md](reports/STRONG_ATTACK_RESULTS.md)
- [TOPOLOGY_FOLLOWUP.md](reports/TOPOLOGY_FOLLOWUP.md)
- [TOPOLOGY_FORMULATION.md](reports/TOPOLOGY_FORMULATION.md)
- [TOPOLOGY_TRAINING_PROTOCOL.md](reports/TOPOLOGY_TRAINING_PROTOCOL.md)

### Frozen experiment protocols and expected results

49 preserved files; filenames retain the original version/date.

- [DEVELOPMENTAL_COUPLING_V1.md](protocols/DEVELOPMENTAL_COUPLING_V1.md)
- [DEVELOPMENTAL_COUPLING_V2.md](protocols/DEVELOPMENTAL_COUPLING_V2.md)
- [DEVELOPMENTAL_COUPLING_V3.md](protocols/DEVELOPMENTAL_COUPLING_V3.md)
- [SCC_ALIGNMENT_CONSOLIDATED_HELDOUT_V1.md](protocols/SCC_ALIGNMENT_CONSOLIDATED_HELDOUT_V1.md)
- [SCC_ALIGNMENT_CONSOLIDATED_REPLAY_V2.md](protocols/SCC_ALIGNMENT_CONSOLIDATED_REPLAY_V2.md)
- [SCC_ARCHITECTURE_PORTFOLIO_V1.md](protocols/SCC_ARCHITECTURE_PORTFOLIO_V1.md)
- [SCC_AUTHORIZED_REPLAY_PROBE_V1.md](protocols/SCC_AUTHORIZED_REPLAY_PROBE_V1.md)
- [SCC_CONSOLIDATED_REPLAY_ABLATION_V1.md](protocols/SCC_CONSOLIDATED_REPLAY_ABLATION_V1.md)
- [SCC_COORDINATE_CONSTRUCTION_V1.md](protocols/SCC_COORDINATE_CONSTRUCTION_V1.md)
- [SCC_DEVELOPMENTAL_DEPENDENCY_AUDIT_V1.md](protocols/SCC_DEVELOPMENTAL_DEPENDENCY_AUDIT_V1.md)
- [SCC_DIAGNOSTICS_V1.md](protocols/SCC_DIAGNOSTICS_V1.md)
- [SCC_DISCRETE_CONSTRUCTION_V1.md](protocols/SCC_DISCRETE_CONSTRUCTION_V1.md)
- [SCC_FULL_GRADIENT_CONTINUATION_V1.md](protocols/SCC_FULL_GRADIENT_CONTINUATION_V1.md)
- [SCC_FULL_GRADIENT_PILOT_V1.md](protocols/SCC_FULL_GRADIENT_PILOT_V1.md)
- [SCC_FUNCTIONAL_BASIS_V1.md](protocols/SCC_FUNCTIONAL_BASIS_V1.md)
- [SCC_LEARNED_BOTTLENECK_SCREEN_V1.md](protocols/SCC_LEARNED_BOTTLENECK_SCREEN_V1.md)
- [SCC_LONG_RECOVERY_COUPLING_V1.md](protocols/SCC_LONG_RECOVERY_COUPLING_V1.md)
- [SCC_MEMORY_FACTORIAL_V1.md](protocols/SCC_MEMORY_FACTORIAL_V1.md)
- [SCC_NEXT_CONSTRUCTION_DIRECTION_2026-09-11.md](protocols/SCC_NEXT_CONSTRUCTION_DIRECTION_2026-09-11.md)
- [SCC_NOR_BASIS_V1.md](protocols/SCC_NOR_BASIS_V1.md)
- [SCC_PERSISTENT_LEARNABILITY_V1.md](protocols/SCC_PERSISTENT_LEARNABILITY_V1.md)
- [SCC_PERSISTENT_OPTIMIZATION_CONTROL_V1.md](protocols/SCC_PERSISTENT_OPTIMIZATION_CONTROL_V1.md)
- [SCC_PERSISTENT_REFERENCE_EXTENSION_V1.md](protocols/SCC_PERSISTENT_REFERENCE_EXTENSION_V1.md)
- [SCC_PERSISTENT_REFERENCE_STABILIZATION_V1.md](protocols/SCC_PERSISTENT_REFERENCE_STABILIZATION_V1.md)
- [SCC_PERSISTENT_REFERENCE_V1.md](protocols/SCC_PERSISTENT_REFERENCE_V1.md)
- [SCC_PILOT_RESOURCE_COMPLETION_V1.md](protocols/SCC_PILOT_RESOURCE_COMPLETION_V1.md)
- [SCC_PROJECTED_CONSTRUCTION_V1.md](protocols/SCC_PROJECTED_CONSTRUCTION_V1.md)
- [SCC_RECOVERED_CAPABILITY_CALIBRATION_V1.md](protocols/SCC_RECOVERED_CAPABILITY_CALIBRATION_V1.md)
- [SCC_RECOVERED_CAPABILITY_CALIBRATION_V2.md](protocols/SCC_RECOVERED_CAPABILITY_CALIBRATION_V2.md)
- [SCC_RECOVERED_CAPABILITY_CALIBRATION_V3.md](protocols/SCC_RECOVERED_CAPABILITY_CALIBRATION_V3.md)
- [SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V1.md](protocols/SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V1.md)
- [SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V2.md](protocols/SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V2.md)
- [SCC_RECOVERED_DEVELOPMENTAL_PILOT_V1.md](protocols/SCC_RECOVERED_DEVELOPMENTAL_PILOT_V1.md)
- [SCC_RECOVERED_DEVELOPMENTAL_PILOT_V2.md](protocols/SCC_RECOVERED_DEVELOPMENTAL_PILOT_V2.md)
- [SCC_RECOVERED_DEVELOPMENTAL_REPLICATIONS_V1.md](protocols/SCC_RECOVERED_DEVELOPMENTAL_REPLICATIONS_V1.md)
- [SCC_RECOVERED_DEVELOPMENTAL_REPLICATIONS_V2.md](protocols/SCC_RECOVERED_DEVELOPMENTAL_REPLICATIONS_V2.md)
- [SCC_RECOVERED_HELDOUT_CHRONOLOGY_2026-09-11.md](protocols/SCC_RECOVERED_HELDOUT_CHRONOLOGY_2026-09-11.md)
- [SCC_RECOVERED_HELDOUT_CONFIRMATION_V1.md](protocols/SCC_RECOVERED_HELDOUT_CONFIRMATION_V1.md)
- [SCC_RECURRENT_STATE_V1.md](protocols/SCC_RECURRENT_STATE_V1.md)
- [SCC_REPAIR_ROUND_V1.md](protocols/SCC_REPAIR_ROUND_V1.md)
- [SCC_SEAM_INTACT_CALIBRATION_V1.md](protocols/SCC_SEAM_INTACT_CALIBRATION_V1.md)
- [SCC_SEAM_POST_QUALIFICATION_V1.md](protocols/SCC_SEAM_POST_QUALIFICATION_V1.md)
- [SCC_SEAM_POST_QUALIFICATION_V2.md](protocols/SCC_SEAM_POST_QUALIFICATION_V2.md)
- [SCC_SELECTIVE_CONSTRUCTION_SCREEN_V1.md](protocols/SCC_SELECTIVE_CONSTRUCTION_SCREEN_V1.md)
- [SCC_SHARED_PREDICATE_V1.md](protocols/SCC_SHARED_PREDICATE_V1.md)
- [SCC_SHARED_READER_V1.md](protocols/SCC_SHARED_READER_V1.md)
- [SCC_TRANSITION_TIMING_V1.md](protocols/SCC_TRANSITION_TIMING_V1.md)
- [SCC_TRANSITION_V1_EXPECTED.json](protocols/SCC_TRANSITION_V1_EXPECTED.json)
- [SCC_VARIED_CORE_COUPLING_V1.md](protocols/SCC_VARIED_CORE_COUPLING_V1.md)

### Artifact readouts, implementation records and readiness documents

59 preserved files; filenames retain the original version/date.

- [scc-bottleneck-implementation-20260911-v1/IMPLEMENTATION.md](artifacts/scc-bottleneck-implementation-20260911-v1/IMPLEMENTATION.md)
- [scc-bottleneck-implementation-20260911-v1/final-protocol.md](artifacts/scc-bottleneck-implementation-20260911-v1/final-protocol.md)
- [scc-bottleneck-readout-20260911-v1/READOUT.md](artifacts/scc-bottleneck-readout-20260911-v1/READOUT.md)
- [scc-charon-and-runtime-20260912-v1/READOUT.md](artifacts/scc-charon-and-runtime-20260912-v1/READOUT.md)
- [scc-charon-integration-20260912-v1/READOUT.md](artifacts/scc-charon-integration-20260912-v1/READOUT.md)
- [scc-combined-status-20260912T212358Z/READOUT.md](artifacts/scc-combined-status-20260912T212358Z/READOUT.md)
- [scc-combined-status-20260912T221236Z/READOUT.md](artifacts/scc-combined-status-20260912T221236Z/READOUT.md)
- [scc-combined-status-20260912T223206Z/READOUT.md](artifacts/scc-combined-status-20260912T223206Z/READOUT.md)
- [scc-construction-status-20260912T063956Z/GMAN_MANUAL.md](artifacts/scc-construction-status-20260912T063956Z/GMAN_MANUAL.md)
- [scc-construction-status-20260912T082846Z/READOUT.md](artifacts/scc-construction-status-20260912T082846Z/READOUT.md)
- [scc-construction-status-20260912T084640Z/READOUT.md](artifacts/scc-construction-status-20260912T084640Z/READOUT.md)
- [scc-construction-status-20260912T085740Z/READOUT.md](artifacts/scc-construction-status-20260912T085740Z/READOUT.md)
- [scc-construction-status-20260912T193327Z/READOUT.md](artifacts/scc-construction-status-20260912T193327Z/READOUT.md)
- [scc-coordinates-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-coordinates-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-dependency-readout-20260911-v1/READOUT.md](artifacts/scc-dependency-readout-20260911-v1/READOUT.md)
- [scc-discrete-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-discrete-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-full-recovery-readout-20260912-v1/READOUT.md](artifacts/scc-full-recovery-readout-20260912-v1/READOUT.md)
- [scc-full-recovery-status-20260912T032902Z/STATUS.md](artifacts/scc-full-recovery-status-20260912T032902Z/STATUS.md)
- [scc-launch-repair-20260912-v1/READOUT.md](artifacts/scc-launch-repair-20260912-v1/READOUT.md)
- [scc-long-coupling-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-long-coupling-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-master-document-20260910-v1/review-notes.md](artifacts/scc-master-document-20260910-v1/review-notes.md)
- [scc-memory-factorial-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-memory-factorial-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-memory-status-20260912T202842Z/READOUT.md](artifacts/scc-memory-status-20260912T202842Z/READOUT.md)
- [scc-persistent-learning-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-persistent-learning-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-persistent-learning-implementation-20260912-v1/STRUCTURAL_NOTES.md](artifacts/scc-persistent-learning-implementation-20260912-v1/STRUCTURAL_NOTES.md)
- [scc-pilot-20260910-v2-publication/literature-positioning.md](artifacts/scc-pilot-20260910-v2-publication/literature-positioning.md)
- [scc-pilot-20260910-v2-publication/report-draft-v2.md](artifacts/scc-pilot-20260910-v2-publication/report-draft-v2.md)
- [scc-pilot-20260910-v2-publication/report-draft.md](artifacts/scc-pilot-20260910-v2-publication/report-draft.md)
- [scc-pilot-20260910-v2-publication/report-template-v3.md](artifacts/scc-pilot-20260910-v2-publication/report-template-v3.md)
- [scc-pilot-20260910-v2-publication/review-guide-draft.md](artifacts/scc-pilot-20260910-v2-publication/review-guide-draft.md)
- [scc-pilot-20260910-v2-readiness/protocol.md](artifacts/scc-pilot-20260910-v2-readiness/protocol.md)
- [scc-pilot-status-20260911-v1/STATUS.md](artifacts/scc-pilot-status-20260911-v1/STATUS.md)
- [scc-portfolio-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-portfolio-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-projected-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-projected-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-recovered-capability-20260910-v1-controls/protocol.md](artifacts/scc-recovered-capability-20260910-v1-controls/protocol.md)
- [scc-recovered-capability-20260910-v1-signal/protocol.md](artifacts/scc-recovered-capability-20260910-v1-signal/protocol.md)
- [scc-recovered-capability-20260910-v2-controls/protocol.md](artifacts/scc-recovered-capability-20260910-v2-controls/protocol.md)
- [scc-recovered-capability-20260910-v2-signal/protocol.md](artifacts/scc-recovered-capability-20260910-v2-signal/protocol.md)
- [scc-recovered-capability-20260910-v3-controls/protocol.md](artifacts/scc-recovered-capability-20260910-v3-controls/protocol.md)
- [scc-recovered-capability-20260910-v3-signal/protocol.md](artifacts/scc-recovered-capability-20260910-v3-signal/protocol.md)
- [scc-recovered-capability-20260910-v3-trajectory/protocol.md](artifacts/scc-recovered-capability-20260910-v3-trajectory/protocol.md)
- [scc-recovered-capability-20260910-v3-trajectory-v2/protocol.md](artifacts/scc-recovered-capability-20260910-v3-trajectory-v2/protocol.md)
- [scc-repair-readout-20260911-partial-v1/READOUT.md](artifacts/scc-repair-readout-20260911-partial-v1/READOUT.md)
- [scc-repair-readout-20260912-v1/READOUT.md](artifacts/scc-repair-readout-20260912-v1/READOUT.md)
- [scc-repair-round-20260911-v1/IMPLEMENTATION.md](artifacts/scc-repair-round-20260911-v1/IMPLEMENTATION.md)
- [scc-research-reset-20260912-v1/READOUT.md](artifacts/scc-research-reset-20260912-v1/READOUT.md)
- [scc-seam-post-20260911-v1-readiness/protocol.md](artifacts/scc-seam-post-20260911-v1-readiness/protocol.md)
- [scc-seam-post-20260911-v2-readiness/protocol.md](artifacts/scc-seam-post-20260911-v2-readiness/protocol.md)
- [scc-selective-readout-20260911-v1/READOUT.md](artifacts/scc-selective-readout-20260911-v1/READOUT.md)
- [scc-status-reconciliation-20260912T224630Z/READOUT.md](artifacts/scc-status-reconciliation-20260912T224630Z/READOUT.md)
- [scc-transition-implementation-20260912-v1/IMPLEMENTATION.md](artifacts/scc-transition-implementation-20260912-v1/IMPLEMENTATION.md)
- [scc-transition-readout-20260912-v1/READOUT.md](artifacts/scc-transition-readout-20260912-v1/READOUT.md)
- [scc-varied-coupling-implementation-20260911-v1/IMPLEMENTATION.md](artifacts/scc-varied-coupling-implementation-20260911-v1/IMPLEMENTATION.md)
- [scc-varied-readout-20260911-v1/READOUT.md](artifacts/scc-varied-readout-20260911-v1/READOUT.md)
- [scc-varied-status-20260911T201902Z/STATUS.md](artifacts/scc-varied-status-20260911T201902Z/STATUS.md)
- [scc-varied-status-20260911T202927Z/STATUS.md](artifacts/scc-varied-status-20260911T202927Z/STATUS.md)
- [scc-varied-status-20260911T204528Z/STATUS.md](artifacts/scc-varied-status-20260911T204528Z/STATUS.md)
- [scc-varied-status-20260911T211212Z/STATUS.md](artifacts/scc-varied-status-20260911T211212Z/STATUS.md)
- [scc-varied-status-20260911T213451Z/STATUS.md](artifacts/scc-varied-status-20260911T213451Z/STATUS.md)

### Earlier shareable deliverables

3 preserved files; filenames retain the original version/date.

- [SCC_Master_Document.docx](deliverables/scc-master-20260910-v1/SCC_Master_Document.docx)
- [SCC_Master_Document.md](deliverables/scc-master-20260910-v1/SCC_Master_Document.md)
- [SCC_Mechanism_Theory_and_Research_Agenda_2026-09-12.docx](reports/SCC_Mechanism_Theory_and_Research_Agenda_2026-09-12.docx)

### Machine records and source navigation

- [Exact current GPU ledger](artifacts/developmental-current-status.json): registered IDs, observations, receipts and result paths; contains earlier completed controls as well as this batch.
- [Code/document catalog](docs/catalog.json): source entry points and preserved protocol/report inventory.
- [Operations guide](docs/OPERATIONS.md): status commands, evidence handling and restoration procedures.
- [Artifact store](artifacts/) and [local runs](runs/): original parents, failures, checkpoints, data, source snapshots, training and prediction logs. These directories are intentionally excluded from source-only transfers.
- [This consolidation's evidence coverage](artifacts/scc-labnotes-20260913-v1/document-coverage.json) and [pre-edit tracked-file hashes](artifacts/scc-labnotes-20260913-v1/before-tracked-sha256.json): audit trail for the backfill.

New chronological entries go **above this index**. Do not create a new report to keep this index growing.
