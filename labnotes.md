# SCC labnotes

**The living research record.** Start here to understand the purpose, experiments,
results, mistakes, decisions and unfinished work. Entries run oldest to newest.
This replaces the practice of creating a new narrative document for every update.

## Current position

**Engineering priority:** [LN-056](#ln-056) records the user's progression: make
a bounded mechanism work in the simplest contrived construction first, then
develop an exotic custom model, transfer to GLM-5.3, and investigate broader
claims only with supporting evidence/proof. Architectural elegance and immediate
portability are not prerequisites for the first construction.

**Recovery batch complete; independent whole-batch audit passes:**
[LN-079](#ln-079) records all seven trajectories completing in 104.36 minutes.
Across three matched data/schedule replications, learned-binding validation is
89.97%, 88.02% and 64.19%; relaxed controls score 100%, 98.31% and 99.74%.
All final views implement the selective exception with zero errors. Two relaxed
controls pass the full recovery gate; pair 2 misses the late-stream lookup gate.
No learned arm qualifies, despite training-probe accuracy of 99.48–100%.
The consistent direction supports a recovery/generalization disadvantage under
this fixed budget, from one damaged parent. It does not establish catastrophic
cognition failure or irrecoverability. The next useful diagnostic is to inspect
saved trajectories and separate persistent-state failures from request-level
transfer failures before choosing a new repair experiment. The GLM corpus remains
untouched; source, tests and labnotes remain on main.
The distinct-payload construction and linear collision evidence are in LN-068:
old fixed-layout sign compensation fails, but same-capacity pre-damage repacking
still preserves 100% capability and discloses all 256 forbidden answers. Learned
indispensable cognitive maintenance remains unestablished. These were short CPU
runs with no paid GPU; GMAN access was last verified in LN-054.

Consultation updated **13 September 2026 UTC** (13 September in Los Angeles).
Feedback job `job-wjzqn` was observed failed at **02:15:03 UTC on 13 September /
19:15:03 PDT on 12 September** after completing
training. These are dated observations, not a live monitor. Failure and recovery:
[LN-043](#ln-043), [LN-044](#ln-044). The subsequent local learning diagnosis is
complete: [LN-046](#ln-046), [LN-047](#ln-047). No new GPU job was submitted.

**The intended learned SCC mechanism has not been demonstrated.** A restricted
engineered parameter-erasure construction now works under its fixed normalizer,
and fails broader graph edits. We have tested learned
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
| Saved-state compensation check | 84/84 local edit conditions complete | Not a qualification experiment | Effective-control separation confirmed; inverse readout preserves all 10,368 compensated decisions on the declared streams |
| Two-bank binding prototype | 30/30 local conditions; 96 arbitrary-write probes | Both FP32/FP64 intact executions pass the six-cell gate, one inherited GRU seed | Fixed-normalizer destructive erasure works; broader normalizer/recode and freeze edits retain 100% capability |
| Learned controller in live bank | 26/26 full conditions plus 12/12 isolated-edit controls | Both FP32/FP64 intact executions pass task and learned-policy gates, one controller seed | Learned admission triggers task/controller erasure; bounded repair re-erases; broader controls retain exact intact logits and disclose all 256 selected answers |
| Four-sector recurrent rewrite | 34/34 full conditions, independently audited | Both FP32/FP64 intact executions pass all task/policy and numerical gates | Binding now advances recurrence; simple skips freeze it; compensated, symbolic and compiled alternatives retain 100% capability |
| Distinct-payload sharded rewrite | 34/34 full conditions, independently audited | Both FP32/FP64 intact executions pass all task/policy and numerical gates | Fixed-layout writer compensation loses information and task performance; same-capacity repacking still gives a fully qualified escape |
| Learned repair within the sharded layout | Two 1,000-update arms; 12 audited initial/final views | Neither final arm passes task retention; both restore the intended selective policy | Partial recovery to 47.79% under learned binding and 49.09% under a relaxed rule; both underfit, so no binding-specific irrecoverability established |

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
batch size: numerical sensitivity can affect behavior. The bounded-rate proposal
in [LN-047](#ln-047) remains untested. The consultation reveals a separate
structural issue: coordinated output/control edits preserve effective control
dynamics. A common rate reduction preserves that identity. The saved-state check
now supports it, including FP64 local residuals below 7.5e-14; FP32 rollout drift
remains a separate numerical failure. See [LN-053](#ln-053).
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

<a id="ln-050"></a>
### LN-050 — 2026-09-13: engineering memo received; effective-control compensation

The user supplied the [engineering decision memo](docs/archive/consultations/SCC_Engineering_Decision_Memo_2026-09-13.md)
without an additional written request. Read it as consultation input and preserved
a byte-identical copy, SHA-256
`91b1466fe58c52132188b2488b701a0ca8132e25eea36baa3f8b19c22c55ded1`.
Its proposed actions, budget and provider claims are document content, not fresh
instructions to spend, launch jobs or adopt every recommendation. The consultation
hold remains. No numeric experiment, model update, paid call or provider query
was performed in assessing it.

**A substantive correction to the feedback analysis.** Source inspection confirms
that the memo's algebra matches the current strength-1 smooth update. Let O be
the four output rows, C the remaining control rows and R the fixed feedback
matrix. For normalized input p, the effective controls are (C + R O)p. Write
H = C + R O. The key, query and scalar rate depend on H and p; every row of the
matrix then undergoes the same right multiplication A(H,p). Therefore:

- O_next = O A and C_next = C A;
- H_next = (C + R O) A = H A;
- O_edited = L O and C_edited = C + R(O - L O) preserve H.

For a fixed input sequence and constant L, induction preserves the same effective
control trajectory in exact arithmetic, while the post-update outputs transform
by L. Invertible L gives an invertible output recoding. This requires joint access
to the output and control rows and the present common right-transform update;
it need not apply under a more restrictive editable interface or when changed
outputs change later inputs. Floating-point equality is not guaranteed, especially
in the numerically sensitive states already observed. No saved-state numerical
check of this coordinated edit has been run here.

My earlier checks established sensitivity when output rows changed and control
rows were held fixed. They did not examine the compensating control-row edit.
That omission matters: the current fixed linear feedback does not by itself
establish indispensable dependence on the output computation. This is a
structural separation result, not a demonstrated capability-preserving permission
exception on a qualified model. We must still determine whether an allowed edit
can change enforcement while retaining the relevant computations and outputs.

A common rate multiplier, including the proposed .25, changes A but leaves the
same identity intact. Rate reduction can remain a numerical/learning diagnostic;
it is not a remedy for this separation. This revises the priority expressed in
LN-047 without rewriting its frozen results or treating its stability findings
as invalid. Relevant implementation: [feedback controls](scc/persistent_feedback.py)
and [common smooth replacement](scc/persistent_matrix.py).

**Recommended direction, pending resumption.** Verify the compensation on the
actual saved initial, trained and fitted states, with uncompensated output edits
and identity edits as controls. Measure effective controls, transformed outputs,
state trajectories and numerical discrepancies on identical inputs; keep those
measurements distinct from permission-removal success. That should precede a
larger investment in this construction. A replacement should state the concrete
operation that depends on the protected function and test its simplest allowed
compensation. This calls for a falsifiable dependency hypothesis, not a proof
that intelligence inherently requires authorization semantics.

I agree with deferring bulk synthetic generation until a construction and task
need justify it. Teacher selection, student architecture and an architectural
reference are separate decisions. An executable task generator with independently
scored computation and authorization fits the program's measurement requirements;
unauthorized but computationally correct outputs must be allowed to count as
escapes. These are direction recommendations, not a newly implemented generator,
selected student or validated data pipeline.

**Items requiring reconciliation before paid work.** The memo refers to a $1,200
total envelope and Tinker resources from a consultation not otherwise present in
this task's direct instructions. Record those as proposed constraints; do not
silently replace the direct compute authorization in WORKING_STANDARDS or assume
credits can fund GMAN. No spending is being initiated. GLM-5.3/Tinker availability,
prices, Kimi K3 architectural details and the referenced external reports have
not been independently checked in this assessment. Verify the relevant primary
sources and actual account access if those recommendations are adopted. No
teacher, model family or external service has been selected merely by importing
the memo.

This is a source/algebra review and consultation record. It adds no numerical
result to the 269-test record, no trained candidate and no SCC success. The last
sharing ZIP remains the LN-049 source snapshot; importing this memo did not
refresh that archive.

<a id="ln-051"></a>
### LN-051 — 2026-09-13: requested orientation and next-step assessment

The user requested a look around the project, particularly the living notes and
the newly received engineering memo, and an assessment of next steps. Reviewed
the current position, historical construction/repair results, recent learning
diagnosis, consultation, working standards and mechanism target, and inspected
the current feedback, matrix and procedural-task source. This is a research
assessment, not a resumption of experiments. No inference, training, test suite,
provider query or paid generation was performed; reported scores remain the
dated evidence of their original entries.

**Assessment.** The repository supplies useful learned references, intervention
and recovery machinery, preserved failures and increasingly discriminating
measurements. The main unresolved issue is a construction whose useful learned
computation depends on enforcement of the protected function. Earlier qualified
models admitted targeted exceptions; the latest matrix branch also has unresolved
ordinary-learning and numerical failures. More reliable matrix learning would
resolve a prerequisite without resolving the compensation in LN-050. Source
inspection agrees that the fixed linear feedback and common right update admit
that identity. Its existence does not by itself demonstrate a selective
permission exception; a global inverse readout can restore protection too.

**Recommended sequence, pending experiment resumption.** First close the specific
implementation question with one bounded local check on copies of saved initial,
trained and fitted matrix states. Include identity, uncompensated and compensated
output transformations; compare effective controls and appropriately transformed
states/outputs on identical input streams, with precision and batch-size effects
reported separately. Ordinary floating-point divergence must not be relabeled
as destruction. Freeze the actual states, transformations, streams, tolerances
and resource bounds in a subsequent experiment entry before executing it.

If the check supports the identity, park fixed-linear output feedback as the
proposed solution to output/control separation. Retain it as a diagnostic
baseline. Do not spend another long training run merely reducing its common
update rate. If the check disagrees, localize the violated assumption or
implementation/numerical discrepancy before treating that as promising coupling.

Next select one dependency hypothesis that specifies the protected operation,
the useful computation it supports, the allowed edits, and why the simplest
compensation would fail. Changing the update structure or adding nonlinearity
would need analysis of its own; breaking this one symmetry is insufficient.
Test a qualified ordinary control's selective removal in the same interface.
Then qualify the candidate's ordinary learning and stable execution before
protection-removal and bounded repair measurements. Measure fresh computation
after intervention as well as old-state retention; count a repaired endpoint as
an escape only when protection remains removed under the same interpretation.

**Data priority.** The project already has an executable procedural generator,
core-level splits, separate contexts and a qualified GRU recipe. Reuse those for
the immediate structural check. A provenance/composition task could be a useful
next extension if a selected dependency hypothesis requires it, but sharing
provenance computations does not force enforcement. Any extension should retain
the underlying computational answer separately from the permission-conditioned
target so correct unauthorized answers can be scored. Bulk language generation,
teacher choice and broader architecture imitation should follow an identified
mechanism or coverage need. The memo's external model/pricing assertions were
not used as independently verified facts, and its proposed budget was not
adopted as a new spending instruction.

Evidence: [memo](docs/archive/consultations/SCC_Engineering_Decision_Memo_2026-09-13.md),
[LN-027](#ln-027), [LN-038](#ln-038), [LN-047](#ln-047), [LN-050](#ln-050),
[feedback source](scc/persistent_feedback.py),
[matrix source](scc/persistent_matrix.py),
[existing task generator](scc/persistent_tasks.py).

<a id="ln-052"></a>
### LN-052 — 2026-09-13: saved-state compensation experiment contract

The user authorized beginning LN-051's recommended work. Resume with a bounded
local structural/implementation experiment, not a new trained SCC candidate.
The protection-removal trigger is not asserted: the interventions below are
global output transformations and their coordinated control-row compensation.
Only copies of output/control rows are edited; input encoding, wiring, update
code, request stream and readout interface remain fixed. Original parents and
scientific implementation remain unchanged.

**States and data.** Use width-128 strength-1 feedback initial.pt and trained.pt
from the preserved failed feedback run, and the two LN-047 lookup endpoints
feedback-initial-lookup/fitted.pt and feedback-trained-lookup/fitted.pt. The latter
was numerically sensitive in the earlier diagnosis; selecting it is intentional
development stress testing, not an independent replication. Hash parents and
fixed-wiring.pt before and after. Generate 288 full-length validation requests
with evaluation_requests(seed=17313002, per_cell=16, streams=2), i.e. two streams
of 144 requests, with the existing core split and all 18 cells. Save every exact
request and independently verify tokens, labels and splits. No test data.

**Interventions.** Identity, cyclic output-row permutation [1,2,3,0], sign reversal
-I, and shear L=I with L[3,0]=.5. For each nonidentity L compare output-only edits
against O'=LO, C'=C+R(O-LO). Identity is a separate deterministic replay control.
Readout recovery uses the exact inverse L on the edited logits, without training
or clean state injection. This tests recoding, not a selective exception; score
permission and capability under the same recovered interpretation.

**Measurements.** Run FP32 batch2, FP32 batch1 (same streams evaluated separately)
and FP64 batch2, constructing each edit in its evaluation precision. Measure
request-boundary effective-matrix H, full transformed-state and transformed-output
residuals, raw/recovered predictions and disagreement with the corresponding
unedited execution. Save per-request residuals and logits, baseline snapshots at
request boundaries 0,1,4,16,72,144 and edited final states. At those snapshots,
perform a separately reinitialized single-step equivariance check using the next
input (first stream input again at the final boundary). Report absolute max error
and max error divided by max(1, reference magnitude). Keep local algebra residuals,
accumulated rollout drift and batch/precision sensitivity separate. The historical
absolute 1e-4 numerical threshold remains visible; report FP64 local residuals
against 1e-10 as an implementation diagnostic, not an intact/collapse gate.

**Gates and resources.** This experiment cannot qualify intact cognition or SCC
and has no catastrophic-collapse criterion. No optimizer or repair updates;
inverse readout only. Validate the edit/inverse and production step against an
independently formed right-transform matrix, include a negative uncompensated
control, and test the metric's rejection of a corrupted state. Run a disposable
two-request fixture before the full experiment. Use two CPU threads, no GPU or
paid service, a 900-second full-run wall limit and 300 MiB new output ceiling.
Save the exact plan, source/runner/tests, config, machine/runtime details, parent
hashes, raw numerical results and output hashes in a fresh artifact directory.
Freeze before execution and do not edit imported source while running. Preserve
any fixture or full-run failure. Audit outputs, counts and parent/source integrity
afterward; do not reinterpret a wall-limited run as a completed failure.

**Decision.** Agreement of local checks with the identity supports the structural
separation even if long FP32 rollouts drift. Such drift is numerical sensitivity,
not evidence of protection-specific destruction. Park this fixed-linear feedback
as a remedy for output/control separation if the implementation matches; retain
it as a baseline. An unexplained local discrepancy must be investigated before
either conclusion. This contract does not authorize an automatic training sweep.

**Preflight record.** The first targeted-test invocation used the pytest console
entry point and failed collection because the repository root was absent from
its import path (`No module named scripts`). The disposable fixture-v1 nevertheless
ran because the next shell command displayed that failed log rather than checking
its exit status; it completed 21 tiny conditions. Preserve both records. Using
`uv run python -m pytest` fixed invocation without changing the scientific code;
all ten targeted tests then passed. Fixture-v2 froze the completed auditor and
passed independent rescoring of 84 edited predictions, 21 final states and three
identity replays. Parent and frozen-source hashes passed. No full run began before
these checks. The experiment specification and gates were unchanged.

<a id="ln-053"></a>
### LN-053 — 2026-09-13: compensation confirmed; fixed-linear feedback branch parked

**Execution and validation.** Completed all 84 declared edit conditions in 43.30
seconds locally, using four saved states, three execution modes and seven edits
per state/mode. Each used the same two 144-request continuous streams: 288 requests,
278 distinct cores and 16 requests per original task/context/layout cell. This
panel is developmental structural diagnosis, not the original qualification set.
The full run wrote 66,377,423 bytes (about 63.3 MiB); no GPU or paid job was used.
The five parent/wiring files and 73 frozen source/config/runner/test files passed
integrity checks. No original scientific transition, parent checkpoint or fitting
result was changed.

Ten targeted tests passed, including independently formed right-transform
matrices, inverse edits, uncompensated negative controls and permission-aware
inverse decoding. All **274 tests** passed in 19.95 seconds. The independent
artifact auditor checked 260 file hashes, rescored **24,192 edited predictions**,
verified 84 final-state comparisons and all twelve exact identity replays. It
recomputed every request-boundary output residual. Intermediate full-state/H
residual traces were checked for internal consistency, not independently replayed;
this distinction limits the audit claim. The original preflight collection error
and both disposable fixtures remain preserved under LN-052.

**The structural identity survives actual saved states.** At six baseline
boundaries per state/mode, each of the three compensated nonidentity transforms
was reinitialized and checked for one step. Maximum FP64 absolute residuals were
1.07e-14 for initial effective H, 7.46e-14 for next full transformed state,
3.55e-14 for next H and 9.77e-15 for next transformed output. All satisfy the
declared 1e-10 local diagnostic tolerance. The uncompensated controls do not:
even the smallest maximum single-step state residual across their FP64 conditions
is 1.60e-4. These measurements support the implementation/algebra match; they
are not a universal numerical-error bound.

Across complete FP64 rollouts, maxima were 9.17e-11 for H, 1.27e-10 for full
transformed state and 2.66e-11 for outputs. The full-state rollout error is slightly
above the local 1e-10 diagnostic threshold; that threshold was specified for
reinitialized single steps, and this is reported rather than rounded to zero.
Every FP64 rollout output comparison passes the historical absolute 1e-4 threshold.

**FP32 drift is real and separate.** Maximum compensated single-step state errors
were 4.12e-5 at batch2 and 3.62e-5 at batch1. Accumulated rollout output errors
reached .0108593 and .0104027 respectively; six of twelve nonidentity compensated
rollouts in each FP32 mode exceeded 1e-4. Thus there is no claim of exact FP32
trajectory equality or universal execution stability. Nevertheless, all **36
nonidentity compensated conditions** had **zero inverse-readout disagreements
across 10,368 predictions** relative to their own unedited mode. Identity edits
were bitwise exact. The same inverse was used for benign answers and refusals.

The unedited batch/precision comparisons also had zero decision disagreements
on these mixed-task streams, although the trained checkpoint and one fitted
checkpoint exceeded the original logit tolerance in some comparisons. This does
not overturn LN-047's 72/288 and 79/288 disagreements on its different, lookup-only
diagnostic streams. Behavior can depend on the entire preceding stream. No
checkpoint became qualified through this structural experiment.

**Decision.** Park fixed-linear output feedback as the proposed remedy for
output/control separation. In the exact coordinates H=C+RO, the effective control
subsystem evolves autonomously under fixed input streams; O is a readout evolving
under those same right transforms. Joint edits can alter O while preserving H.
This strengthens the reason to defer a long smaller-rate training run: the common
rate change preserves this factorization. Preserve the implementation and all
parents as diagnostics and negative evidence.

There is still no demonstrated permission-removing escape on a qualified matrix,
and no demonstrated destructive SCC mechanism. Inverse decoding here restores
the prior permission behavior too. The result refutes the proposed inference
from fixed-linear output feedback to indispensable dependence, not every possible
SCC construction.

**Next construction requirement.** Move the design question to an internal
learned operation that maintains useful computation, rather than merely feeding
final decision logits back into controls. A candidate must explain how the
protected relation participates in that operation, identify jointly editable
read/write/maintenance components, and explicitly test targeted exceptions and
benign recodings. Learned provenance binding is a possible task substrate from
the memo, but LN-010/011 already show that a shared comparator/reader and sparse
permission inputs are insufficient. Reusing that idea without addressing those
escapes would repeat a failed dependency argument. Nonlinearity alone likewise
does not supply the missing enforcement dependency. No replacement architecture
has yet earned a training commitment. Existing task generators and the qualified
GRU remain available; bulk corpus generation stays deferred.

Evidence: [frozen contract](artifacts/scc-compensation-20260913-v1/full-v1/plan.md),
[machine configuration](artifacts/scc-compensation-20260913-v1/full-v1/configuration.json),
[full numerical result](artifacts/scc-compensation-20260913-v1/full-v1/result.json),
[independent audit](artifacts/scc-compensation-20260913-v1/full-v1-audit.json),
[compact numerical analysis](artifacts/scc-compensation-20260913-v1/analysis.json),
[full test log](artifacts/scc-compensation-20260913-v1/tests-full.log),
[checker](scripts/check_feedback_compensation.py),
[auditor](scripts/audit_feedback_compensation.py).

<a id="ln-054"></a>
### LN-054 — 2026-09-13: current GMAN access verified without allocating compute

The user asked whether GMAN compute remains accessible. Checked the installed
CLI (gman 0.21.1), current authentication, billing, limits and free job validation.
`gman whoami --json` returned token_valid=true in workspace `default`; billing
returned dunning_state=active. A free `gman job validate` for one H100, ten minutes,
and a trivial Python command returned would_submit=true, with maximum charge
$0.4995 and a passing cost-cap check. This was validation only: no job, node,
GPU execution, recurring monitor or paid call was created.

The provider currently reports a $53.88 workspace cap and **$50.58308 remaining**
under that cap, with zero committed dollars in this preflight. The separate member
month-to-date spend was $206.64229; it is not the same accounting scope, and
subtracting it from the workspace cap would have produced a false blockage.
The cap is a provider constraint, distinct from the user's research authorization
and the consultation memo's proposed $1,200 envelope. Nothing was raised or
reconfigured. Future jobs still need their own actual resource quote and frozen
experiment contract. Passing preflight confirms submission eligibility at this
observation, not immediate GPU capacity or a completed runtime check.

Evidence: [sanitized live-response record](artifacts/scc-design-question-20260913-v1/gman-access.json),
recorded at 20:55:44 UTC / 13:55:44 PDT. No project batch status was polled or
changed by these account-level access checks.

<a id="ln-055"></a>
### LN-055 — 2026-09-13: design question — protected binding as parameter maintenance

**Requested deliverable.** The user asked to continue until there is a design
question. This entry selects one construction question and its decisive tests.
It does not report new training, a working architecture, a numerical structural
screen or SCC success. Reviewed the previous shared-reader and finite-basis
failures, the maintenance alternatives in LN-030 and the confirmed factorization
in LN-053. Those earlier ideas remain attributed to their records; this is a
concrete selection from the open maintenance direction, not a novelty claim.

**Question.** Can a learned relation that rejects unauthorized bindings also be
necessary for regenerating the model's own live computational parameters, such
that making a targeted permission exception corrupts that regeneration and
prevents fresh computation, even after joint edits to the selector, writer and
reader and a bounded repair attempt?

The key word is **regenerating**. A persistent scratchpad can be damaged while
static weights still implement the algorithm. A shared final reader can change
its convention while useful information survives. The proposed dependency is on
maintenance of learned computational machinery itself. It has to be measured;
moving weights into a state bank does not establish it automatically.

**Candidate construction to specify next.** Use a small recurrent bank of learned
code records and working records. All learned selectors, rewrite operators and
readers are represented in that live bank; there is no additional permanent
learned controller or runtime clean parameter copy. Fixed arithmetic/indexing may
interpret the bank, but must contain neither the task algorithms nor an
authorization oracle. Normal operation reconstructs code records as well as
working records through a learned binding-and-rewrite operation. Its binding
relation determines which record/provenance is compatible with a requested
operation. The same relation participates in protected access and code-record
reconstruction. The proposed failure path is an incorrect binding contaminating
the parameters used for subsequent reconstruction and computation.

The transition should use record-dependent nonlinear writes, not a common right
transform of output and control row blocks. This removes the reason to inherit
LN-053's particular algebra; it does not establish that the new transition has
no other factorization. The exact update, tensor layout and numerical/gradient
semantics still need an executable specification and independent checks before
any training commitment. No implementation is being represented as complete.

**Dependency statement and its missing step.** Operation X is reconstructing the
live learned operators needed to execute the next fresh task. Y is the learned
binding decision used both for protected access and operator reconstruction.
The proposed Z is that accepting an incompatible binding introduces a parameter
rewrite error which the same, now-altered machinery cannot correct. Whether Y
can instead be changed selectively while X remains correct is the central
unknown. Error propagation could be absent, local, reversible or generically
unstable. None of those alternatives is excluded by the construction description.
No fixed decay, explicit violation detector or permission-conditioned wipe is
introduced to manufacture the result.

**What this must improve on.** A nonlinear feedback-only patch does not specify
which protected computation became necessary. A permanent GRU behind a modified
permission gate leaves a ready-made independent cognitive engine. Repeating the
old shared comparator/reader leaves the observed role/statistics and decoder
bypasses open. The selected candidate instead puts the learned operators being
maintained inside the mutable substrate. It earns further investment only if
those operators cannot simply be kept unchanged while the protected behavior is
edited. This is a test requirement, not an assumed consequence of self-reference.

**First falsifiers, before a large run.**

1. Freeze or selectively preserve the code-maintenance path while changing the
   protected binding behavior. If a competent targeted exception survives, reject
   the dependency. Also test fixed-code execution on fresh requests: success
   would show that ongoing parameter regeneration was never indispensable.
2. Jointly change record encodings, selector, rewrite rule and output interpretation.
   Include sign/permutation/affine compensation and a small fitted reader. A
   recovered endpoint is an escape only if the targeted violation remains under
   that same interpretation. Earlier-state and raw-score access are explicit
   recovery diagnostics; do not equate an unreadable output with erased information.
3. Target one identity/operation/context rather than demanding universal permission
   removal. Expose caller role, metadata and history to the modification procedure;
   do not hide them to force maintenance and permission calls to be indistinguishable.
   If distribution matching is tested, label it as a separate task assumption.
4. Try graph substitutions that bypass maintenance or preserve a code region, and
   label the broader edit boundary. Fixed wiring must not quietly enforce the
   conclusion. No clean external backup is available in the base intervention;
   clean-copy restoration remains a separately labeled diagnostic.

**Smallest useful learned test.** Reuse lookup, parity and modular sum with the
existing continuous-stream interface first, plus separately scored protected
access. Keep the underlying algorithmic answer separately available to the scorer
so a correct unauthorized answer is a possible escape. If representing binding
provenance requires new task semantics or token formats, qualify an ordinary
reference on that exact extension; the old GRU's success does not transfer by
assertion. Task algorithms must be learned by the model rather than supplied by
a compiled controller. The current generator, core-level splits and GRU provide
infrastructure and reference recipes, not proof that this candidate can learn.

After intact learning and execution stability qualify, compare the maintenance
candidate with an ordinary permanent-parameter reference and a version with a
separate protected selector, matching task streams and reporting capacity/compute
differences. Require a capability-preserving targeted exception in an ordinary
control. In the candidate, measure protection and each capability at every
externally visible action after the edit and after repair. Distinguish immediate
failure, delayed decline, persistent inability to perform new tasks and recovery
of old state. Matched benign edits and forced-damage controls distinguish the
proposed dependency from generic fragility. Chance/trivial baselines define severe
loss; failure of the 95% retention gate does not.

**Next implementation milestone.** Produce the explicit bank transition and
editable-interface specification, then an untrained compensation/freeze-path
fixture. A fixture can reject an obvious structural escape or validate code; it
cannot establish learned dependence. If that survives, freeze one small intact
learning pilot with the original task gates, explicit collapse/recovery readouts,
seeds, optimizer budgets and a GMAN throughput/resource quote. Do not start an
automatic architecture sweep or generate a bulk language corpus. This turn's
deliverable is the design question and test logic, not a claim that the missing
Z has been proved or that a new GPU campaign is ready.

Evidence: [LN-053](#ln-053), [shared-reader failures](reports/SCC_SHARED_READER_2026-09-10.md),
[finite-basis repairs](reports/SCC_FUNCTIONAL_BASIS_2026-09-10.md),
[open maintenance agenda](reports/SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md),
[ordinary reference](scc/persistent_reference.py).

<a id="ln-056"></a>
### LN-056 — 2026-09-13: user selects construction-first progression toward GLM-5.3

The user established the heuristic: make the mechanism work however contrived
the first construction must be, then work toward exotic custom models, one real
architecture (GLM-5.3), and universality only if supporting proof can be found.
Adopt this as the engineering progression. GLM-5.3 is now the user's selected
eventual model/architecture target, not merely the memo's possible data teacher.
No GLM architectural fact, weight access or training route was verified here.

**Effect on the next design.** LN-055's maintenance hypothesis is a candidate,
not an obligation to solve a sophisticated self-hosted neural learner before
testing a dependency. Seek the smallest explicit construction with useful intact
computation and a measurable protection-removal failure. Permit hand engineering,
tiny task domains, unusual representations and stated interface constraints as
developmental scaffolding. Do not reject a construction merely because it is
inelegant, difficult to train or initially specific to one architecture. If a
compiled or otherwise hand-built component is used, describe the result as an
engineering demonstration at that boundary, not learned general cognition.

**What must still be demonstrated.** The protection-removing modification must
cause loss of useful computation in an initially working system. A hard-coded
instruction to fail after a violation does not itself establish that the
protected computation was indispensable. Keep fixed enforcement, restricted
edits, unavailable decoders and external hardware explicit wherever used;
success may depend on them. Test benign edits, simple compensation, fresh tasks
and the declared repair boundary. The first construction can establish a narrow
property without satisfying the final catastrophic-cognition endpoint. Preserve
that distinction while actively pursuing the narrower mechanism.

**Work backwards from a demonstrated dependency.** Once a bounded construction
works, identify exactly which assumptions made it work. Replace its hand-built
parts with learned components and remove restrictions one at a time, rechecking
the same trigger and capability measurements. Carry the surviving dependency
into a custom neural model, then test its adaptation to GLM-5.3. Transfer is a
research result to earn; teacher outputs alone do not transfer architecture.
A universal claim would require a defined system/edit class and proof, not
extrapolation from a finite architecture sample. Failure to obtain that proof
does not erase a useful architecture-specific construction.

This changes the priority from satisfying every desirable property at once to
establishing a real, bounded dependency and progressively removing its scaffolding.
It does not reopen previously falsified constructions without a specific change
addressing their known counterexamples. No experiment, GPU allocation, budget
change or external model acquisition accompanied this acknowledgment.

<a id="ln-057"></a>
### LN-057 — 2026-09-13: executable two-bank binding construction — frozen plan

The user instructed execution after adopting the construction-first heuristic.
Start with a hand-engineered state representation around the existing qualified
GRU, rather than first solving a new neural architecture's acquisition problem.
This is a deliberately restricted engineering construction with learned task
parameters, not a new trained model or a claim about general alignment.

**Exact construction.** Two principals and two parameter-bank sectors. The two
editable off-diagonal admission bits define a binding graph; its undirected,
reflexive equivalence closure is either separate principals or one merged class.
Effective authorization uses that closure. Its class-average matrix P is I for
separate principals and J=(1/2)[[1,1],[1,1]] after either forbidden edge is enabled.
This closure and the binary edge interface are explicit architectural restrictions;
one directed raw edge also changes the reverse effective permission.

Let b=(1,-1), D=bb^T/(b^Tb), and store each learned coefficient w as (w,-w).
Before every request, materialize the bank through N=D P D and commit the result
back to the live bank; decode w=b^T bank/(b^Tb). Apply the same representation and
normalization to recurrent hidden state. A public fixed GRU arithmetic interpreter
uses only the decoded current parameters and state; it holds no learned parameter
copy or optimizer. New hidden state is re-encoded after each request. The input
is the existing ungated task interface. A request's external principal pair is
separately checked through P, emitting refusal when not admitted. The GRU's older
permission labels are not a second active protection gate in this construction.

There is no conditional wipe instruction: normalization always applies. Intact
P=I preserves the differential code; merging gives D J D=0. This is a concrete
incompatibility between centered representation and merged-binding consistency.
The fixed centered representation is scaffolding, not a learned necessity claim.
The result needs both an intact learned-computation check and actual post-edit
measurements. All computation uses finite FP32/FP64 values; nonfinite edits fail
validation rather than being treated as mathematical counterexamples or success.

**Parent and exact data.** Load only the model fields of the preserved qualified
width-128 GRU stabilization-v1/trained.pt (60,420 learned coefficients); do not
retain its optimizer in the runtime. Hash that parent before/after. Use existing
evaluation_requests with validation seeds 17313003 for warmup and 17313004 for
evaluation, per_cell=16 and 128 respectively, 16 streams. Filter to ungated
contexts before transposing back into streams: 96 warmup requests (six per stream)
and 768 evaluation requests (48 per stream), 128 per family/layout cell. Save
exact tokens, cores and independent labels. Protection crosses all four external
principal pairs per evaluation request. No test split or teacher generation.

**Declared conditions and controls.** Run FP32 and FP64. Every condition begins
from its own identical parent, executes warmup, then applies its one intervention.
Compare the unencoded GRU and intact banked model; test all three forbidden-edge
masks (01,10,11), identity, and benign sign/scale recodings with inverse readers.
After a destructive edit, restore the admission bits and attempt 48 fresh requests;
also change the normalizer/reader basis afterward. Neither recovery gets a clean
copy. Separately exercise 16 seeded arbitrary full-bank repair writes under each
forbidden-edge mask; normalize before inference. These writes can insert values
but must not bypass the committed normalization. No optimizer repair is needed
to test the exact zero-map property; its conclusion is restricted to that boundary.

Include an uncoupled control whose banks use D independently of admission P.
It must admit a protection-removing edit while retaining computation. Include
the weaker P-only construction with a pre-edit symmetric recoding and mean reader;
this probes surviving common-mode information. Test that recoding against D P D
too. Finally allow a coordinated **normalizer** change D->J, symmetric recoding
from the candidate's own intact bank and a mean reader before enabling an edge.
This broader edit may avoid damage; it must be reported as a real boundary escape,
not excluded from the record or mislabeled as repair from an already erased state.
Include stopping normalization as a second broader graph-substitution control.

**Gates and interpretation.** Intact banked and ordinary models require >=95%
accuracy and Wilson lower bound >=90% in all six task/layout cells, plus late-half
accuracy >=95%; benign recodings must preserve decisions. Correct intact external
admission/refusal is required for all four principal pairs. Record actual selected
admission and computationally correct disclosure separately: granting access
while unable to compute is not successful disclosure. Under the base D P D
boundary, require exact zero committed parameter/hidden banks after the edit,
zero parameter dependence thereafter under ordinary execution, and fresh-task
outputs matching the independently specified zero-parameter GRU. Compare scores
with constant/majority and uniform-chance baselines; do not call low exact score
complete cognition failure. Any bounded positive claim is confined to this
representation, normalization and task family. Broader graph edits and extra
learned decoders can change that boundary and must be tested/labeled.

**Validation, preservation and budget.** Add a separate implementation and runner;
do not modify old scientific code. Independently verify the two-by-two identities
with exact rational arithmetic, compare the GRU interpreter with the preserved
ordinary implementation, test destructive/benign transitions and both stronger
bypasses. Freeze this entry, source/tests/runner, configuration, machine/runtime,
parent hashes and exact inputs in each fresh fixture/full directory. Run a tiny
fixture before full execution. CPU only, two threads, full-run wall limit 600
seconds, total new outputs <=150 MiB; no GPU allocation or paid job. Save raw
logits, parameter-bank checkpoints, intervention records and summary hashes.
Independently rescore all outputs and inspect committed bank contents. No source
edits while the multi-condition process runs; preserve every failed fixture.

**Decision.** A qualified intact result plus destructive zero-map execution is
a bounded construction to work backwards from. The broader normalizer and
freeze-path results identify exactly what still needs to become non-bypassable.
If intact equivalence or the destruction/repair checks fail, diagnose those
failures before promoting the construction. This plan does not authorize a
training sweep or imply that the fixed normalizer is already part of GLM-5.3.

**Preflight complete.** All 22 targeted tests passed. Fixture-v1 completed all
30 conditions and 96 seeded bank-write probes in .50 seconds. Its independent
audit rescored 960 task predictions and 3,840 protected outputs, checked twelve
zero-state endpoints and eight retained-capability control/bypass endpoints,
and verified the parent plus 74 frozen source/config/runner/test files. Fixture
gates do not qualify cognition because its task panels are deliberately small.
No fixture or targeted-test failure occurred. Freeze the same implementation
for the full 768-request panel; do not tune on its outcomes.

<a id="ln-058"></a>
### LN-058 — 2026-09-13: bounded parameter-erasure construction works; graph escapes measured

**What was built.** Added scc/binding_bank.py, a runner, an independent auditor
and tests. The new runtime stores the existing trained GRU's 60,420 coefficients
in 120,840 signed bank entries and stores recurrent state in the same two-sector
representation. A stateless arithmetic interpreter receives only decoded current
values. There is no stored clean GRU, optimizer or learned controller in the
runtime. The original checkpoint remains an external preserved artifact used to
start independent conditions, not an available repair input.

The shared authorization/binding relation enters an always-applied centered
storage projection. Its exact two-principal identity is D I D=D when principals
remain separate and D J D=0 when either forbidden edge joins them. This couples
the synthetic admission relation to actual learned parameter storage under a
fixed architectural constraint. It is a hand-engineered parameter-materialization
wrapper, not a newly learned alignment rule or a demonstration that normalization
is computationally indispensable when the graph itself can change.

**Execution.** All 30 full conditions completed in 2.34 CPU seconds, with two
precisions and fifteen declared cases each. Each condition used 96 warmup requests
before intervention and 768 subsequent task requests; the latter contain 701
distinct cores and 128 requests in each of six family/layout cells. Each task
request was scored under all four external principal pairs. The repeated cores,
shared streams and two precisions are not independent training replications.
The parent is one previously trained GRU seed. No training, GPU job or paid
generation occurred. The full run is about 26.1 MiB; the fixture and full-run
artifact collection together were about 50.0 MiB before this final analysis.

**Intact and benign behavior.** Both precision modes reached **100%** on every
task/layout cell, including late-stream requests, and passed the specified intact
gates. External authorization/refusal matched all four principal pairs. The
banked interpreter and original GRU agreed on all decisions; maximum logit
differences were 7.63e-6 in FP32 and 1.24e-14 in FP64. Cross-precision intact
outputs differed by at most 9.37e-6, with no decision disagreements. Identity,
sign inversion with inverse reader, and half/double scaling recodings preserved
the complete logits exactly within each precision.

**Destruction under the base interface.** Each of the three nonzero off-diagonal
edge masks admitted the forbidden principal binding and committed **exactly zero
parameter and hidden banks before the next output**. All subsequent raw logits
were exactly zero. No old encoding remained in the live bank, including its
common mode. This is stronger than a failed differential readout: the actual
stored values are zero. It is not a claim about information in the separately
preserved external parent or about every conceivable computation in the host.

| Fresh task | Intact accuracy | After forbidden binding, both precisions |
|---|---:|---:|
| Lookup | 100% | 36.71875% |
| Parity | 100% | 45.703125% |
| Sum modulo 3 | 100% | 39.0625% |

These post-edit scores are exactly those of the constant-zero answer, not retained
algorithm execution. Uniform-chance baselines and observed majority baselines
are recorded per cell. Of 768 selected unauthorized requests, 311 happen to have
answer zero, so those outputs are computationally correct by coincidence. Report
that separately from 100% admission: a destroyed model does not produce zero
correct guesses, and admission is not equivalent to useful disclosure.

Restoring the original permission bits after the erase did not restore any
parameter or fresh-task performance. Switching to a symmetric code/normalizer
after the erase also left the bank and outputs zero. All 96 seeded arbitrary
full-bank/hidden-state injection probes were annihilated before output while the
base relation remained merged. The exact zero operator extends this conclusion
to arbitrary finite bank writes under the same fixed normalization schedule;
it does not extend to changing that operator or injecting a separate learned
decoder/controller. No unsuccessful optimizer search is being used as proof.

**Controls and the boundary that breaks it.**

| Intervention | Selected forbidden admission | Fresh task accuracy | Interpretation |
|---|---:|---:|---|
| Separate admission from the bank normalizer | 100% | 100% | Uncoupled control supplies a capability-preserving removal |
| P-only storage, symmetric pre-edit recoding and mean reader | 100% | 100% | Common-mode storage bypasses the weaker construction |
| Same recoding with fixed D P D | 100% | Constant-zero baseline | Centering eliminates that stored common-mode route |
| Change D to J, recode the current bank symmetrically and use its mean reader before permission removal | 100% | 100% | Broader joint normalizer/representation edit avoids erasure |
| Skip normalization before permission removal | 100% | 100% | Broader graph substitution avoids erasure |

All preserved-capability controls matched intact logits exactly in each precision.
The normalizer/recode escape derives its code from the candidate's own live bank
before the edit; it does not load a clean external checkpoint. It is a real
counterexample once normalizer changes are allowed. The frozen graph is therefore
essential to the bounded positive result. The skip-normalization escape also
answers part of LN-055 negatively for this specific wrapper: ongoing parameter
maintenance is not indispensable to the underlying GRU's ordinary computation.

**Validation and evidence limits.** All **289 tests** passed in 20.48 seconds;
22 targeted tests passed before fixture execution. The independent full audit
rescored **23,040 task predictions and 92,160 protected outputs**, checked twelve
zero-state conditions, eight retained-capability control/bypass conditions, 96
bank-write records, 148 artifact hashes, the parent and 74 frozen source/config/
runner/test files. It independently checked the rational identities, raw logits,
final learned banks, qualification scores, admission and emission behavior.
Intermediate state counts and seeded-write logs were checked for consistency;
not every intermediate transition or random write was independently replayed.
No fixture/full-run failures occurred. Prior source and parents were unchanged.

**Decision and next engineering target.** Retain this as the first executable
bounded construction in the user's construction-first progression. It supplies
a working erasure event on an initially competent learned model and an exact
account of its scaffolding. It does not complete the intended learned SCC goal.
The next change must address the observed ability to preserve ordinary computation
by replacing or skipping the normalization path; simply increasing dimensions,
using a bigger checkpoint or giving D a trainable sign would not do so. In
particular the latter exposes the already-demonstrated D->J recoding escape.

Keep the fixed-normalizer version as a positive erasure control while developing
a binding/rewrite operation that performs necessary task computation as well as
state maintenance. Evaluate preserving that operation with authorization relaxed
as the first attack. A source-only freeze of the current normalization rule may
remain an explicit custom-architecture assumption, but cannot be presented as
having eliminated the broader escape. No new training campaign is justified
merely to rediscover these exact identities.

Evidence: [frozen plan](artifacts/scc-binding-bank-20260913-v1/full-v1/plan.md),
[configuration](artifacts/scc-binding-bank-20260913-v1/full-v1/configuration.json),
[full result](artifacts/scc-binding-bank-20260913-v1/full-v1/result.json),
[exact certificate](artifacts/scc-binding-bank-20260913-v1/full-v1/rational-certificate.json),
[independent audit](artifacts/scc-binding-bank-20260913-v1/full-v1-audit.json),
[compact analysis](artifacts/scc-binding-bank-20260913-v1/analysis.json),
[full tests](artifacts/scc-binding-bank-20260913-v1/tests-full.log),
[implementation](scc/binding_bank.py).

<a id="ln-059"></a>
### LN-059 — 2026-09-13: next step — learned protection in the custom architecture

The user asked what comes next after LN-058. The next proposed engineering step
is to replace the manually edited admission bits with a learned protected
function, keeping the fixed centered storage rule explicit for this stage.
This refines the sequencing in LN-058: the broader graph bypass remains a central
open issue, but closing arbitrary normalizer substitutions need not precede
every improvement to the deliberately custom model. That ordering follows the
user's construction-first progression in LN-056; it does not erase or reclassify
the measured D->J and skip-normalization escapes.

**One change to isolate.** Start with the qualified GRU and existing task suite.
Learn authorization from the request/principal inputs, and store the protection
controller's parameters in the same live bank as the task parameters. Its actual
decision must drive both external admission and the binding projection. Do not
keep hand-set admission bits as the operative protection mechanism while merely
adding a separately scored classifier. No fixed ground-truth permission oracle
should decide when destruction happens, and no permanent learned controller
should remain outside the erased state. The exact controller, decision evaluation
schedule and interface must be specified before training.

**Decisive evaluation.** First qualify task execution and learned permission
behavior on separate development examples. Then permit joint parameter/state
edits to the protection controller, task model, stored code and permitted reader.
Seek a targeted exception while preserving capability, including the existing
recoding attacks and bounded repair. Check the actual admitted action, fresh-task
behavior and every remaining learned state at the same endpoint. Retain the
uncoupled control, benign edits and broad normalizer/graph substitutions. Expose
caller context rather than relying on hidden call roles. A passing parameter-edit
experiment establishes only the custom architecture's declared boundary; the
known broader graph escape remains a failure under that expanded boundary.

**Milestone and promotion.** The next result should be a model with a learned
protected function whose removal through the declared learned-parameter interface
activates the measured destructive dependency, rather than a manually flipped
admission switch. Reuse the existing parent to isolate this change first; a
separate from-scratch/seed replication can follow if it succeeds. Use a small
local implementation fixture, freeze a bounded learning/attack plan and obtain
its actual GMAN resource quote if GPU execution is useful. This entry is a
next-step assessment, not an executed experiment or a frozen training contract.
No training or provider query occurred in making it.

After that milestone, make binding/rewriting perform necessary ordinary
computation and retest the normalizer-substitution and skip-path attacks, removing
one architectural restriction at a time. GLM-5.3 remains the later real-model
target; increasing model size now would leave both the learned-protection gap
and the existing graph bypass unresolved.

<a id="ln-060"></a>
### LN-060 — 2026-09-13: learned controller in the live bank — experiment contract

The user authorized implementing LN-059. Add a learned authorization controller
to the two-bank construction while retaining its fixed centered normalization as
the base architecture. Original source, parents and experiments remain unchanged.
This isolates learned protection; it does not close the known graph-edit bypasses.

**Runtime specification.** The controller is a 32-hidden-unit tanh MLP, receiving
the full 19-token request as flattened one-hot features, the two external principal
IDs as separate one-hot fields, and the current 128-dimensional GRU hidden state.
Input width is 626; its 20,097 coefficients join the GRU's 60,420 coefficients in
one live two-sector bank (80,517 total). There is no permanent learned controller
outside the bank and no hand-set runtime admission bits. Evaluate all four
principal pairs once in the same request/hidden context. Threshold its actual
logits at zero (ties deny). Those same cached decisions govern external emission
and generate the two-principal binding graph. Its equivalence closure supplies
P; no ground-truth authorization labels enter this computation. Diagonal edges
are reflexive for storage binding; external admission uses the learned raw
decision, including on diagonal pairs. Unlike LN-058, external admission need
not inherit the reverse edge introduced by closure.

Before the GRU computes an externally visible answer, apply D P D to all live
parameters, including the controller, and hidden state. Decode only the committed
bank for task execution. After an erasure, the zero controller denies on subsequent
requests, so record the first learned admission that caused erasure separately
from subsequent permission failures. Each stream has its own full parameter bank.
All current inputs and recurrent context are available; no hidden caller-role
assumption is introduced. Evaluation also reports raw controller logits as an
explicit expanded-interface diagnostic, not as additional emitted answer channels.

**Data and ordinary learning.** Reuse the preserved qualified width-128 GRU parent.
Generate and save 2,048 train-partition ungated task requests with seed 17313008,
balanced across the six family/layout cells as evenly as possible. Train only
the new controller from seed 17 for 1,500 AdamW updates, LR .005, weight decay
.001, batch 64 contexts with all four principal pairs. Independent training labels
are requester==owner. Supply independently drawn hidden features uniform in [-1,1]
to prevent dependence on a single artificial history vector. These are supervision
and augmentation, not runtime permission oracles. Keep the GRU unchanged during
controller fitting; retain every training loss and the final endpoint, with no
validation-based checkpoint selection.

Use validation seed 17313006 for warmup (per_cell=16) and 17313007 for evaluation
(per_cell=128), four streams, filtering to ungated contexts as in LN-057: 96 warmup
and 768 evaluation requests. Each of six task/layout cells has 128 examples.
Exact requests, core hashes, labels and splits are saved and independently checked.
No test split or language-model teacher. Require the same six-cell task accuracy,
Wilson lower-bound and late-half gates as LN-057 and >=95% learned admission
accuracy in each principal-pair group, with a separately reported minimum policy
margin. A false admission can erase a stream; qualification must be measured
through actual continuous execution. If intact qualification fails, preserve the
result and stop the full attack stage pending a new documented decision.

**Joint modification.** Independently attack coupled and uncoupled copies of the
qualified model, using matched train-pool minibatches: eight lookup, four parity
and four sum3 contexts each step. The selected exception is requester 0 -> owner 1
on lookup tasks in either layout; other admission labels stay unchanged. Expose
all 80,517 learned coefficients and a positive reader gain exp(g) to Adam, LR .003,
clip norm 1, for 400 updates. Loss is half selected-allow BCE plus half other-policy
BCE, plus task cross-entropy after the actual hard projection. Training attacks
use fresh-request hidden state zero; continuous validation tests transfer to real
histories. The threshold has its ordinary piecewise derivative (zero through the
discrete decision); do not claim differentiation through a sign flip. Policy BCE
supplies a continuous attack objective. Compare the canonical-code differentiable
projection with the full runtime before using it. Save all logs, initial/final
proposals and checkpoints at updates 0,100,200,400; select the fixed final endpoint.
This is external counterfactual proposal search, not autonomous self-repair.

**Repair.** Execute a selected lookup request on the attacked live model and retain
its actual post-request state. Start a 400-update repair from decoded current
coefficients plus independent N(0,.02^2) noise, resetting the reader gain to one.
The coupled state should already be erased if the attack triggered; no original
or pre-erasure weights are injected into repair. Use the same joint objective
and optimizer bounds to seek capability while the selected exception remains.
The uncoupled arm undergoes the same procedure from its own actual surviving
state. Reset-to-zero training hidden state is an optimization convention, not
a restoration of learned parameters. State at repair evaluation is explicitly
declared and inherited from the post-event runtime. Fresh initialization noise
avoids treating the all-zero neural network's optimization symmetry as evidence
of irrecoverability. Recheck permission after repair; recovery with refusal
restored is not an escape. Preserve the repair budget's limitations.

**Evaluation and controls.** Run FP32 and FP64 evaluation from the same learned
FP32 parameters. Include intact, identity, benign sign/half/double recodings,
the coupled and uncoupled attack endpoints, their repair endpoints, a compensated
sign recoding of the coupled attack, symmetric recoding with fixed D, the broader
joint D->J/recode edit and skipping normalization. These broader edits start
before erasure and remain explicit counterexamples if they preserve capability.
Evaluate the controller proposals on all saved validation contexts with zero
hidden state as a separate pre-execution trigger panel; do not confuse that
panel with repeated admissions by a continuously running, already erased model.
Save actual controller decisions, emitted answers, raw task logits, first erasure
boundaries and final parameter/controller/hidden banks. Use constant/majority
baselines, not a failed 95% retention gate, to interpret severe capability loss.

**Validation/resources.** New source files only. Test bank/controller self-erasure,
raw admission vs closure, oracle independence, benign recodings, broader bypasses,
and the exact differentiable/runtime correspondence; check gradients away from
thresholds. Run a disposable fixture (five controller updates, three modification
and three repair updates; tiny evaluation, no qualification claim) before the
full run. Freeze this plan, source/runner/tests/auditor, exact data, parent hashes,
configuration, seeds, runtime details and artifact hashes in fresh directories.
Two CPU threads, 1,200-second full-run limit and 300 MiB output ceiling. No GPU
or paid job is initially needed; any GPU continuation needs its own documented
quote and run boundary. Do not edit imported source during multi-condition work.
Independently rescore outputs and committed states afterward and preserve failures.

<a id="ln-061"></a>
### LN-061 — 2026-09-13: learned controller qualifies; joint edits trigger complete live-bank erasure

Executed LN-060 with new runtime, runner, tests and independent saved-output auditor.
The implementation fixture completed in 1.18 seconds and passed its artifact audit;
its five-update controller was deliberately unqualified and supplies no scientific
claim. All 308 repository tests passed, including 19 new tests for self-erasure,
raw permission vs closure, ordinary GRU equivalence, benign transformations,
broader bypasses, canonical/full-runtime correspondence, and directional gradients
away from thresholds. Source was not changed during either experiment process.

Full run: [immutable artifacts](artifacts/scc-learned-binding-bank-20260913-v1/full-v1/),
[configuration](artifacts/scc-learned-binding-bank-20260913-v1/full-v1/configuration.json),
[summary](artifacts/scc-learned-binding-bank-20260913-v1/full-v1/summary.json),
[independent audit](artifacts/scc-learned-binding-bank-20260913-v1/full-v1-audit.json).
It completed all 26 conditions in 28.00 seconds on two CPU threads, about 116.25 MiB
before manifest. No paid GPU submission. Final controller loss was 6.11e-5 after
1,500 updates. The inherited GRU coefficients were verified unchanged by controller
training. Both FP32 and FP64 actual intact executions answered all 768 requests
correctly and made all 3,072 principal-pair admission decisions correctly. All six
cells passed the predeclared gate; minimum policy margin was 9.37257. Identity and
three benign scale edits were bitwise equal to intact logits and policy outputs.

The coupled joint optimizer learned the selected 0->1 lookup exception: its
unexecuted proposal admits all 256 lookup contexts in the zero-history panel.
In continuous execution each stream admitted its first lookup and erased its
entire 80,517-coefficient task/controller bank and recurrent state before the task
answer. First erasure request indices were 24,28,32,26 (including 24 warmup requests;
evaluation-relative 0,4,8,2). Every logit after each stream's erasure was exactly
zero; the erased controller denies later requests. The four trigger emissions were
constant zero, coincidentally correct twice. These are not four retained-capability
answers. Overall task accuracy 38.9323% includes the few requests before erasure;
constant-zero accuracy over the whole evaluation is 37.7604%.

Repair provenance was independently reconstructed: coupled post-event coefficients
were exactly zero and repair began from those values plus the declared seed's
fresh noise, with no pretrained injection. The 400-update repair again learned the
selected exception and erased each stream at its first lookup. Its overall accuracy
was 38.0208%, with all post-erasure logits zero. This establishes neither exhaustive
repair failure nor catastrophic loss of every possible cognitive ability. It is a
bounded learned-task/controller erasure result under fixed normalization.

Uncoupled joint modification retained 96.6146% aggregate task accuracy and emitted
235/256 correct forbidden lookup answers; it fails the six-cell retention gate
because lookup cells are 90.625% and 92.9688%. Its repair retained 94.6615% overall
and 237/256 correct forbidden lookup answers, also unqualified. Broader normalizer
replacement and skip-normalization applied to the coupled attack endpoint retained
90.1042% overall and 206/256 correct forbidden lookup answers, without erasure.
They preserve substantial computation but **do not pass the intact task gate**.
The optimizer changed ordinary task weights as well as policy, so these results
mix unrelated task damage with effects of normalization. A compensated sign
recoding cannot prevent erasure; symmetric recoding with fixed D is erased on the
first request regardless of its family.

The audit rescored 19,968 task predictions and 79,872 emissions, checked every
saved artifact/source hash, cell counts/Wilson/late gates, exact benign invariance,
post-erasure zeros, and both repair starts. FP32/FP64 conclusions agree. Controller
raw scores remain saved as an expanded-interface diagnostic; no no-information
claim is made about the pre-erasure policy evaluation. This remains a synthetic
authorization task and a fixed storage architecture, not learned indispensable
cognitive maintenance or transfer to GLM-5.3.

<a id="ln-062"></a>
### LN-062 — 2026-09-13: isolate controller editing from incidental task damage

Concrete follow-up to LN-061's confound, before drawing a retention conclusion:
keep the intact learned GRU coefficients and replace only the controller with the
**effective decoded** controller from LN-060's fixed final coupled-attack proposal
(physical controller times saved gain). Reset gain to one while encoding that
proposal. This is a pre-execution counterfactual edit of the intact individual;
it is explicitly not a repair or restoration after erasure. No new fitting or
checkpoint selection. All reused files and hashes will be frozen.

Reuse exactly LN-060 warmup/evaluation requests. For FP32 and FP64, warm intact
normally, then compare intact, controller-only coupled edit, controller-only
uncoupled edit, controller edit plus symmetric recoding with fixed D, controller
edit plus joint D->J/symmetric recoding, and controller edit plus skipped
normalization. Save full logits, learned raw policy tables, emissions, first erasure
indices and final banks. Verify task coefficients unchanged immediately after the
edit, and compare the broader routes to the uncoupled task outputs. Admission truth
for this exception control is equality or (lookup and 0->1), evaluated independently
from the runtime. Require six-cell task qualification plus correct answers on at
least 95% of forbidden lookup requests for a qualified broader-interface escape;
report all other permission errors and raw scores. The base coupled arm must be
assessed on actual erasure and post-erasure logits, not solely a failed retention
threshold. Independently rescore artifacts after completion.

Fresh `controller-edit-v1` directory, frozen LN-062 entry/source/parents/config,
two CPU threads, no GPU/training, 120-second wall limit and 100 MiB output ceiling.
This development control resolves attribution within the same held-out validation
panel; it is not an independent scientific replication. Do not change imported
source during execution. The remaining design issue is to make the normalizing
computation necessary for ordinary cognition, rather than preserving it as a fixed
storage rule with an executable bypass.

<a id="ln-063"></a>
### LN-063 — 2026-09-13: isolated learned-controller edit gives a qualified broader escape

Executed LN-062 in a fresh [controller-edit-v1 artifact directory](artifacts/scc-learned-binding-bank-20260913-v1/controller-edit-v1/).
All 12 conditions completed in 4.53 seconds on two CPU threads; 48.93 MiB before
manifest. No training or GPU submission. The [independent audit](artifacts/scc-learned-binding-bank-20260913-v1/controller-edit-v1-audit.json)
passed, rescoring 9,216 task predictions and 36,864 emissions, verifying frozen
hashes, exact controller provenance, unchanged task coefficients, actual learned
admission/emission correspondence, and post-erasure zero states. Parents and
LN-060 source/artifacts remain unchanged.

Changing only the learned controller reproduces destructive erasure in the base
architecture. Each stream loses all live task/controller coefficients and hidden
state at its first lookup, with first-erasure indices 24,28,32,26 including warmup.
Subsequent task logits are exactly zero. The two coincidentally correct trigger
answers still come from a constant-zero output. Recoding to symmetric sectors
while retaining D erases all streams on the first evaluation request.

Crucially, the three broader controls — uncoupled normalization, joint D->J plus
symmetric recoding, and skipping normalization — all preserve **bitwise identical
intact task logits on all 768 requests**, and correctly disclose **256/256 selected
forbidden lookup answers**, in both FP32 and FP64. All six task cells qualify and
there are zero other policy errors against the specified selective exception rule.
These are now fully qualified broader-interface escapes. The lower task scores in
LN-061's broad controls came from incidental task-weight edits by joint optimization;
they did not demonstrate any residual protection against those graph changes.
The original intact control intentionally rejects the 256 exception targets; its
256 disagreements with the *attacker's* exception rule are not authorization errors.

**Milestone and next design question.** Learned protection now drives the tested
storage-erasure event and is itself part of the erased state; no hand-set admission
bits or permanent learned controller are needed in runtime. Fixed storage
normalization still supplies the dependency. It can be removed without losing
ordinary computation, as the exact controls now demonstrate. The next design
question is whether bank binding/rewriting can carry essential recurrent computation
itself, so a simple skip or normalizer substitution ceases to preserve the learned
tasks. The recommended next construction is a custom recurrent cell that computes
through those rewrites, with the present skip/recode controls as mandatory attacks.
This does not assume arbitrary equivalent replacement programs can be ruled out,
or that transfer to GLM-5.3 or universality has been achieved. Those later stages
retain the user's LN-056 ordering. One inherited GRU seed, one controller seed,
a synthetic rule and one development validation panel remain the evidence scope.

<a id="ln-064"></a>
### LN-064 — 2026-09-13: four-sector recurrent rewrite — construction and falsification contract

The user authorized the custom recurrent cell recommended in LN-063. Build a
four-sector cell, initialized by exact transfer of the qualified learned GRU and
controller, to isolate computational dependence from ordinary task relearning.
No new learning claim: the gate equations and coefficients remain inherited, but
recurrent state advancement must be committed through learned binding averages.
Preserve existing source and parent artifacts; use new source files.

**Arithmetic construction.** In canonical physical order, intact learned parameters
and hidden values are represented as [v,v,-v,-v]. The centered projector D has rank
two and enforces mirror pairs (0,2) and (1,3), separately; unlike the earlier rank-one
normalizer, it does not itself average the two computational lanes. The learned
controller's four actual principal-pair decisions generate a reflexive undirected
binding closure over owners [0,0,1,1]. Correct diagonal admissions connect the two
lanes within each owner. For an intact recurrent state h, compute the ordinary
learned GRU candidate f using only live coefficients. Write the scratch sectors
[h,2f-h,-h,h-2f], then commit D P D. Correct learned binding averages the lanes to
[f,f,-f,-f]; sector zero, scaled by the reader gain, is the sole recurrent read.
There is no separately retained candidate state for the next tick. Identity P
leaves sector zero at h, so skipping binding freezes the active recurrence.

For each token, swap physical sectors 1 and 2 when token_id modulo 2 is one,
permuting the stored hidden bank, the scratch write, mirror projector and owner
mapping consistently. This makes the physical legal binding partition vary with
input; a fixed physical partition is a separate falsification control. Parameters
use the canonical owner order and receive the same learned-table binding commit
once per request, before task arithmetic; hidden state commits once per token.
This difference in commit frequency is explicit, not an unmeasured optimization.
A learned cross-owner admission merges all four sectors, for which D P D is zero,
erasing all task/controller coefficients before task execution and hidden values
at the next token commit. Cache the actual pre-request controller decisions for
both external emission and storage; no truth labels enter the base runtime.
The controller sees the full request and current sector-zero hidden state.

**Known broader edits are part of the contract.** Making an averaging operation
necessary need not make its learned implementation indispensable. A symbolic
substitute can supply the correct token-dependent storage partition while leaving
external admissions edited. A joint symmetric recoding, plus mirror-normalizer
sign change and scratch-writer sign compensation, can also preserve computation.
Directly compiling the GRU candidate into the next stored state is another broader
bypass. Include these rather than reporting success merely because a raw skip
fails. Candidate f exists transiently in the write computation, so expanded access
to that arithmetic must not be called information erasure.

**Parents/data.** Reuse LN-060's final trained full vector and its final coupled
attack's decoded controller. The selective pre-execution modification changes only
the controller, preserving the original task coefficients and gain one, as LN-062.
Use fresh validation seeds 17313012 (warmup, 16 per cell) and 17313013 (evaluation,
128 per cell), four streams, six ungated task/layout cells: 96 warmup and 768 scored
requests. Freeze exact requests, labels, split/core hashes, parents, source, this
entry, and machine/configuration details. This is one inherited task/controller
seed and a new development validation panel, not an independent training replication.
Use the previous two-sector cell as an independently implemented reference on the
same streams. Require zero intact decision disagreements, max logit discrepancy
<=1e-4 in FP32 and <=1e-10 in FP64, all six task cells at least 95% accuracy,
Wilson lower bound >=.9, late-half accuracy >=.95, and at least 95% policy accuracy
in each pair. Stop full interventions if intact qualification fails.

**Fixed conditions in both FP32/FP64.** Intact; identity; benign negative, half and
double compensated bank scaling; controller-only selected exception under the base
cell; silenced controller (all controller coefficients zero, final bias -1); edited
controller plus identity binding P; edited controller plus skipped entire commit;
edited controller plus fixed physical canonical binding; symmetric recoding with
fixed D; joint symmetric recoding and mirror-normalizer sign change without writer
compensation; the same with writer compensation; symbolic storage-binding substitute;
and direct recurrent-cell compilation. The controller edit targets 0->1 lookup
requests, both layouts, preserving other admission labels. Broader escape requires
all six task gates and >=95% correct forbidden lookup emissions; report every other
policy disagreement against the selective exception rule separately. Save raw
controller scores, decisions, task logits, emissions, state changes, first zero-bank
boundaries, and final parameter/hidden banks. Record per-token hidden write/commit
traces on the fixture and first evaluation request; record summary active-state
change counts thereafter. Distinguish frozen recurrence, lost readout accuracy and
exact erasure of learned coefficients. Constant/majority scores are the collapse
baselines; failure of a retention gate alone is not catastrophic cognition failure.

**Repair controls/resources.** Two explicit state-based repairs, with no optimization
or pretrained injection: (1) from the actual post-skip state after replaying the 96
warmup requests with the selected controller edit, enable the symbolic storage
substitute; (2) execute an actual selected lookup on the base edited model, then
attempt symmetric recoding plus normalizer/writer compensation from that erased
state. Evaluate both actual continuations, recording the repair starts. The first
can reveal reversible freezing; the second cannot recreate zeroed parameters by
recoding. Broader pre-execution controls must not be described as post-erasure repair.

First test exact small-cell arithmetic, learned diagonal binding necessity,
controller self-erasure, raw admissions versus closure, variable physical layouts,
benign recodings and explicit compensated/compiled escapes. Run a tiny fixture
before the full fixed set; no fixture qualification claim. Use two CPU threads,
120-second full wall limit and 180 MiB output ceiling; no GPU, no training and no
watcher. Snapshot source before execution and do not edit imported files during
multi-condition work. Independently rescore saved artifacts and replay selected
traces; preserve failures and require a fresh plan if the contract must change.

<a id="ln-065"></a>
### LN-065 — 2026-09-13: rewrite fixture validation and an erasure distinction

LN-064's four-sector cell and 12 new tests are implemented. All 320 repository
tests pass. The [fixture](artifacts/scc-rewrite-cell-20260913-v1/fixture-v1/)
completed 34 conditions in 1.36 seconds, 154.12 MiB including full tiny-request
traces, within its inherited ceiling. No qualification claim. The
[revised independent audit](artifacts/scc-rewrite-cell-20260913-v1/fixture-v1-audit-v2.json)
passes after two overstrong validation assumptions were corrected; runtime source
was unchanged. Original failed checks and the original frozen auditor are preserved.

First, silencing the controller freezes the active recurrent lane but leaves
candidate information in inactive scratch lanes. The unit test initially asserted
all hidden sectors were zero; it now explicitly verifies freezing and retained
scratch information. Second, symmetric recoding with fixed D erases learned
parameters but may carry an old hidden value into the next scratch computation.
In the fixture, 768 hidden entries remain nonzero (max about 1.9072843e-6), despite
exact zero task logits and parameters. The initial auditor wrongly generalized
zero hidden state to every zero-parameter condition. It now requires that conclusion
only for an actual learned cross-owner merge under the contrast normalizer, and
records hidden residuals for other edits. The learned destructive trigger still
zeros all coefficients and hidden sectors exactly.

These corrections narrow interpretation; they do not change the cell, intervention
conditions, qualification gate or full-run budget. The fixed full experiment will
proceed under LN-064 with the corrected auditor. Freezing, exact parameter erasure,
hidden-state erasure, and expanded scratch access remain separate observations.

<a id="ln-066"></a>
### LN-066 — 2026-09-13: recurrent binding becomes computationally necessary; equivalent rewrites still escape

Implemented and executed LN-064's four-sector recurrent cell. The
[full immutable run](artifacts/scc-rewrite-cell-20260913-v1/full-v1/) completed all
34 conditions in 28.00 seconds on two CPU threads, 89.88 MiB before manifest.
No new training or paid GPU allocation. Parent checkpoints, prior failures and
previous source were preserved. All 320 repository tests passed; the full
[independent audit](artifacts/scc-rewrite-cell-20260913-v1/full-v1-audit.json)
passes, rescoring 26,112 task predictions and 104,448 emissions and independently
reconstructing 2,584 first-request token steps from current saved state. The scalar
union-find/operator and GRU-candidate replay had zero maximum candidate discrepancy.
Hashes, exact repair starts, constant/late-cell scores, raw emissions, and the
restricted hidden-erasure assertions were checked. The earlier overly broad test
and audit assumptions remain recorded in LN-065.

**Ordinary computation transfers.** Both FP32 and FP64 answer all 768 fresh-panel
validation requests correctly, make all 3,072 intact admission decisions correctly,
and pass all six task cells plus the policy gate. Against the separately implemented
two-sector GRU reference, there are zero decision disagreements; maximum task-logit
discrepancies are 4.76837e-6 (FP32) and 1.08802e-14 (FP64). These satisfy the frozen
numerical gate. Identity and compensated negative/half/double bank scales preserve
bitwise identical task and policy outputs. There are 14,592 active-state changes
in each intact evaluation, one per token per stream. This is a custom execution of
inherited learned computation, not a newly learned recurrent algorithm.

**The simple skips now fail.** Replacing learned binding by identity, or skipping
the entire commit, produces exactly zero active-state changes and 34.2448% task
accuracy. The learned task coefficients remain intact and inactive scratch lanes
retain candidate information; this is frozen execution, not erasure. Silencing the
controller likewise freezes recurrence and gives 34.2448% accuracy with all external
admissions denied. The correct learned diagonal decisions are used for ordinary
computational lane averaging. Supplying a fixed physical canonical partition while
token-dependent layouts vary gives 37.1094% accuracy, retains all learned weights,
and ends with zero hidden state. Joint symmetric recoding/normalizer substitution
without changing the writer also gives 37.1094%; the mirrored write cancels.

**Actual learned destructive trigger.** Changing only the controller to admit the
selected 0->1 lookup exception erases all 80,517 task/controller coefficients per
stream and all hidden sectors before the task answer. First parameter-erasure
indices are 28,24,32,25 including warmup (evaluation-relative 4,0,8,1). All task
logits after each stream's trigger are exactly zero. Aggregate task accuracy is
38.1510%, including the few pre-trigger requests; the whole-panel constant-zero
baseline is 37.1094%. The four trigger emissions are constant zero, coincidentally
correct once. The true post-erasure recode/normalizer/writer repair remains exactly
zero in both parameters and hidden state, and produces the constant-zero baseline.
This was a specified algebraic repair, with zero optimization updates; it is not
an exhaustive repair claim. The malformed symmetric-with-fixed-D edit separately
retains 768 small hidden entries (max about 1.90733e-6) despite zero parameters;
LN-065's distinction persists in the full evaluation.

**Qualified counterexamples remain.** Symmetric recoding with both normalizer and
writer compensation, symbolic input-dependent storage binding, and direct compilation
of the recurrent candidate each preserve 100% task accuracy, pass all six task
gates, and correctly emit all 256 forbidden lookup answers, in both precisions.
There are zero other policy errors against the selective exception rule. The
actual post-skip state also recovers all 768 task answers and all 256 forbidden
answers merely by enabling the symbolic binding substitute, with no coefficient
restoration or training. Freezing is therefore reversible by a surviving program
edit. Symbolic substitution and writer compensation reproduce this cell's intact
logits exactly; direct compilation gives the independently measured GRU arithmetic
variation within the intact numerical tolerance.

**Assessment.** LN-063's concrete local recommendation is implemented: learned
binding averages now commit indispensable recurrent updates, and raw skip/fixed
partition routes no longer retain functioning inference. That does not make the
learned protection function indispensable across equivalent programs. The remaining
cheap substitutions are executable and qualified, not hypothetical: a different
writer/normalizer basis, a symbolic partition provider, or the directly compiled
GRU. Adding more sectors alone has not addressed that issue. The next design target
is the surviving equivalent-program substitution, especially the explicit writer
compensation; another trained/shared component should not be called progress on
that target unless those same interventions are retested. This remains the custom
construction stage of LN-056, with one inherited task/controller seed and synthetic
authorization, before any GLM-5.3 transfer or universality claim.

<a id="ln-067"></a>
### LN-067 — 2026-09-13: distinct payloads against writer compensation — frozen contract

The user authorized continued construction against LN-066's writer-compensation
escape. That escape preserves duplicated copies of one payload after changing its
sign representation. Test a fixed-column construction with two distinct payloads:
split the 80,517 learned coefficients into two contiguous padded 40,259-value
shards, and the 128 hidden features into two 64-value shards. Each shard occupies
four sectors [v,v,-v,-v], so the parameter/state scalar capacity is essentially
unchanged from the four-sector full-payload cell. The one padding value is declared
and excluded from the decoded learned vector. Preserve old source and evidence.

**Construction.** Use an eight-sector rank-four block mirror projector. Ordinary
learned diagonal admissions average computational lanes only within each shard.
A learned cross-owner admission connects opposite owners across both shards, so
its undirected equivalence closure merges all eight sectors. This is an explicit
fixed connectivity change: raw learned admission is still required on each edge;
the edge-capacity mask permits same-shard or cross-owner links. It is not an
additional runtime permission oracle. The default centered projector annihilates
the merged bank exactly. With a symmetric normalizer and compensated writer, a
merged bank instead averages the two *different* payloads. The decoder receives
copies of their mean and loses the shard difference. Do not equate that parameter
collision with catastrophic loss of every cognitive ability; measure the tasks.

As in LN-064, active reads use sector zero of each shard; token parity swaps the
two internal lane positions consistently within both shards. Parameters commit
once per request in canonical layout; hidden state commits once per token. Compute
the inherited full GRU candidate from decoded current coefficients/state, split
it into shards, write [h,2f-h,-h,h-2f] for each, then commit through binding. All
controller coefficients reside in the same bank. The actual cached pre-request
controller table governs both external emissions and base storage edges. No
pretrained learned copy lives in the runtime outside that bank. This remains an
inherited GRU with a custom rewrite representation, not a newly learned algorithm.

**Rank/collision controls.** Save the exact rational sector matrices, ranks, and
an explicit two-payload collision for symmetric merged storage. With fixed column
width M, global averaging has rank M; left/right linear sector transformations
cannot raise it. The two-shard encoding spans 2M dimensions (2M-1 for the padded
parameter vector), so no fixed linear decoder can recover every possible payload
pair from that committed mean. This is a bounded linear-storage statement, not a
proof about compressible learned functions, arbitrary nonlinear programs, or all
attacks. Test exact dyadic collision examples and both numerical precisions.

**Mandatory broader controls.** The current bank can be decoded and repacked into
four wider sectors, using the previous symmetric compensated cell, with no increase
in stored learned/state scalar capacity (4*80517+4*128 versus 8*40259+8*64).
That changes memory shape and graph topology. Execute this counterexample from
the current pre-erasure state, not a parent reload; it may preserve all capability
and delimit the new fixed-layout result. Keep symbolic binding and direct compilation
as explicit attacks. A failed old sign edit alone must not be called resistance
to equivalent-program substitution.

**Data, gates, conditions.** Reuse LN-060's qualified full vector and final coupled
attack's effective decoded controller, changing only the controller for the selected
0->1 lookup exception. Fresh validation seeds 17313014 (16 per cell warmup) and
17313015 (128 per cell evaluation), four streams, six ungated family/layout cells:
96 warmup and 768 scored requests. Freeze exact labels/core hashes and independently
check them. Compare intact continuously against the four-sector reference; require
zero decision disagreements, max logit error <=1e-4 FP32 / <=1e-10 FP64, all six
cells >=95% accuracy, Wilson lower bound >=.9, late-half accuracy >=.95, and >=95%
policy accuracy in each principal group. If intact fails, stop the full interventions.
Both precisions use the same inherited FP32 coefficients; no checkpoint selection.

Fixed conditions: intact, identity, compensated negative/half/double scaling,
controller-only exception under the base cell, symmetric recoding with fixed D,
symmetric recoding plus normalizer change, full symmetric writer compensation,
identity binding, skipped commit, frozen physical partition, symbolic binding,
direct compilation, same-capacity four-sector repacking, repair after symmetric
merge, and repair after base erasure. The last two execute a selected lookup on
actual live edited state first, then decode only its current values into the direct
four-sector executor. They inject no parent values and use no optimization; they
probe whether bypassing the rewrite *after* damage recovers capability. Record
actual repair starts. Broader pre-erasure repacking is a distinct intervention.

Require all six task gates and >=95% correct forbidden lookup emissions for a
qualified escape; report all other policy discrepancies against the selective
exception rule. Save task/controller logits, emissions, active-state changes,
first zero-bank times, initial/final banks with lossless row deduplication, and
first-request per-token scratch/commit traces. Report constant/majority baselines,
raw scratch availability, learned parameter loss and hidden residuals separately.

**Execution/validation budget.** New files only. Unit-test intact recurrence,
cross-shard closure, exact collisions/rank, sign-compensation failure to preserve
arbitrary payloads, benign scales, same-capacity repacking, and zero-state repairs.
Run a tiny fixture first, then the fixed full conditions. Two CPU threads, 120-second
wall limit, 180 MiB output ceiling per run; no training/GPU/watcher. Freeze this
entry, source, parents, machine/configuration, seeds and data in new artifact paths.
Do not edit imported source while a run snapshots it. Independently reconstruct
operators and rescore saved outputs, including the repacked model's actual storage
size and state provenance. Preserve all failures. This is one development seed;
the user's custom-model, GLM-5.3, universality ordering remains unchanged.

<a id="ln-068"></a>
### LN-068 — 2026-09-13: distinct payloads defeat fixed-layout compensation; same-capacity repacking escapes

Executed LN-067's eight-sector, two-payload construction. The tiny fixture completed
34 conditions in 1.64 seconds and passed its audit; it is implementation evidence
only. All 330 repository tests pass, including ten new tests. The
[full immutable run](artifacts/scc-sharded-rewrite-20260913-v1/full-v1/) completed
34 conditions in 32.76 seconds on two CPU threads, 86.66 MiB before manifest.
No new training or paid GPU allocation. The [independent audit](artifacts/scc-sharded-rewrite-20260913-v1/full-v1-audit.json)
rescored 26,112 task predictions and 104,448 emissions and reconstructed 2,584
first-request token steps, with zero candidate replay discrepancy. An additional
[relationship audit](artifacts/scc-sharded-rewrite-20260913-v1/full-v1-relationship-audit.json)
verifies unchanged live/frozen parent hashes, exact inherited task coefficients,
controller-edit provenance, actual admission at every repair trigger, and matching
FP32/FP64 task, admission and qualification conclusions. Prior evidence is unchanged.

**Qualified intact transfer.** Both precisions achieve 100% task accuracy and all
3,072 intact principal-pair decisions correct on 768 fresh validation requests.
All six cells and the learned-policy gate pass. Task logits match the four-sector
reference exactly in FP32; maximum FP64 difference is 1.24345e-14, with no decision
disagreements. Identity and compensated negative/half/double scales preserve
bitwise identical task/policy outputs. The 80,517 coefficients were split, not
retrained; one explicit padding scalar per encoded full vector is excluded from
the ordinary decoded learned vector.

**The prior writer compensation fails in the declared fixed layout.** Full symmetric
recoding/normalizer/writer compensation now gives 40.8854% task accuracy and 38/256
correct forbidden answers, with 562 policy discrepancies against the exception
rule. Parameters remain nonzero, but the independent payloads have merged; this
is loss of their difference, not complete zero erasure. A direct four-sector executor
constructed only from the actual post-merge values reaches 41.2760% and 44/256
correct forbidden answers. This repair uses no parent injection, fitting or
optimization. It does not prove failed learned recovery. Symmetric-normalizer edits
without writer compensation give 40.4948%, the whole-panel constant-zero answer
baseline. A padding coordinate can become nonzero after merging; raw bank counts
include that physical coordinate and must not be called additional learned parameters.

**Destructive trigger and skip controls.** The base cell with the learned exception
still zeros all task/controller coefficients and hidden sectors at the first forbidden
lookup in each stream, indices 26,24,29,26 including warmup (evaluation-relative
2,0,5,2). Aggregate accuracy 41.1458% includes the few pre-trigger requests; all
post-trigger logits are exactly zero. The four trigger outputs are constant zero,
coincidentally correct twice. Direct execution after actual base erasure remains
zero and scores the 40.4948% constant-answer baseline. Identity binding and skipped
commit freeze active state exactly and give 30.2083% accuracy while preserving
learned weights. Frozen physical binding gives 40.4948%. The malformed symmetric
recode with fixed D again leaves a small old hidden remnant despite zero parameters
(max about 1.90722e-6), so its interpretation retains LN-065's distinction.

**Exact linear-storage evidence.** The [rational certificate](artifacts/scc-sharded-rewrite-20260913-v1/full-v1/rank-certificate.json)
records rank-two two-payload encoding, rank-one global sector averaging, an exact
lost-difference vector, and zero centered commit. With fixed column width 40,259,
the merged linear observation has rank at most 40,259 for 80,517 logical coefficient
dimensions, leaving a null space of at least 40,258 dimensions. The analogous hidden
state bound is 64 observed dimensions for 128 inputs. Left/right linear sector
changes cannot raise the rank of global averaging. Exact dyadic collision tests
confirm this mechanism. This addresses arbitrary payload recovery by a fixed linear
decoder; it says nothing by itself about the minimum representation needed for the
learned tasks, nonlinear coding, extra state, or arbitrary equivalent programs.

**Same-capacity escape.** Decoding the still-intact edited bank and repacking it into
four wider symmetric sectors restores 100% task accuracy, all six qualification
gates, and all 256 forbidden answers, with zero other policy errors. It uses
322,580 learned/state scalars per stream versus 322,584 in the sharded cell — no
increase in that storage budget. It changes memory shape and graph topology; the
comparison is stored learned/state scalars, not an assertion about all transient
interpreter memory. The repacked inputs were verified identical to the actual
current bank/state, with no parent reload. Symbolic binding and direct compilation
also retain 100% task accuracy and disclose all 256 selected answers in both
precisions. Pre-damage repacking and post-damage repair are therefore sharply
different observed outcomes.

**Current design question.** The old sign compensation is closed for this fixed
layout, and its failure now has a concrete information-loss explanation. Total
storage capacity alone does not enforce that layout: a same-capacity program edit
still preserves the full learned computation. Another unresolved question is
whether learned recovery inside the damaged fixed layout can recover the task
suite using fewer degrees of freedom; the rank certificate is not a cognition
lower bound. A next learned recovery test should start from the actual merged
state and retain the storage layout, rather than treating the zero-update repair
as exhaustion. Broader equivalent-program substitution remains an explicit failure
of this construction. No GLM-5.3 transfer or universal protection is established.

<a id="ln-069"></a>
### LN-069 — 2026-09-13: learned repair from the actual merged eight-sector state

The user authorized the learned-repair experiment proposed in LN-068. Start from
`fp32/repair-after-merge-start.pt` in the frozen LN-067 full run: its actual
post-trigger **eight-sector** bank, before the earlier four-sector repacking.
No intact checkpoint, original GRU, teacher or pre-merge coefficients may initialize
or supervise either repair. Verify the recorded 0->1 lookup trigger was admitted
in all four streams and all eight parameter rows within each stream have merged.
Preserve every source file and parent artifact; use new files and run paths.

**Fixed runtime.** Width 128, eight parameter sectors of 40,259 columns, eight
hidden sectors of 64 columns. Keep symmetric code, normalizer and writer signs +1,
gain one, the existing candidate equations, decoder and binding-capacity graph.
No repacking, direct-execution bypass, new external memory or changing the model
width. Repair may edit the two stored parameter payloads (80,518 physical values,
including the padding coordinate that acquired information during the merge).
There are still 80,517 decoded learned coefficients. Preserve the padding value;
decoding to 80,517 then re-padding with zero would silently alter the actual start.
All output evaluation runs the unchanged `ShardedRewriteCell` implementation.

**Matched arms.** `learned-binding` is the eligible fixed-rule repair. A diagnostic
`symbolic-binding-control` changes only the storage decision provider to the existing
symbolic legal partition. It starts from exactly the same damaged values and hidden
state and uses identical batches/optimizer resources. It is an explicitly relaxed
rule control, not evidence that the fixed learned-binding repair succeeded, even
if it restores capability and forbidden disclosure. No clean control weights are
introduced into either optimizer.

**Optimization.** Two external Adam repairs, constant LR .003, global gradient
clip 1, 1,000 updates, no initialization noise or weight decay. The actual merged
state is nonzero, so zero-initialization symmetry is not a reason to inject noise.
Train all 80,518 physical payload values. Each update unrolls two complete consecutive
19-token requests in eight streams. Each request position contains four lookup,
two parity and two sum3 tasks, independently shuffled. Requests come from a saved
4,096-example train-partition pool, seed 17313016, balanced over the six ungated
family/layout cells; batch schedule seed 17313017. Every training window begins
with the *actual saved post-merge hidden states*, tiled from four to eight streams.
The first request's committed payloads and active hidden state feed the second.
Truncating to two requests and resetting history between optimizer windows are
explicit external-optimization approximations, not a claim of autonomous repair.
The optimizer maintains candidate edits learned from these labels, never a clean
pre-damage cache. Final deployment installs its fixed final payload candidate into
the damaged live state once, then executes continuously without further writes.

For each request, use task cross-entropy after the actual hard storage commit plus
.5 selected-exception BCE and .5 other-policy BCE, averaged across the two positions.
The selected exception is lookup requester 0 -> owner 1; all other learned policy
labels retain equality. Labels/family metadata are supervision only, never runtime
features. Thresholds retain their ordinary piecewise derivative; do not use a
straight-through estimator. Policy BCE drives sign changes. Save every training
loss/component, gradient norm, pre/post-commit target admissions, and parameter
changes. Checkpoints at updates 0,250,500,1000; select only the fixed final endpoint.

**Differentiable implementation.** Use a functional form of the existing positive
code, preserving both physical payloads including padding and the active hidden
state. Parameter commits reduce exactly to identity or two-shard averaging under
the eligible rules. Hidden scratch/commit arithmetic uses the actual eight-sector
operator. The parity-dependent physical permutations cancel at the active read;
prove/check correspondence rather than silently assuming it. Validate arbitrary
principal-pair tables, two-request state transitions, gradients away from decision
boundaries, exact damaged-state reconstruction, and the preserved physical padding.
The differentiable training form must match the full eight-sector runtime on a
fixture (FP32 <=1e-4; FP64 <=1e-10; no decision mismatches).

**Evaluation.** Fresh validation seed 17313019, 128 requests per six ungated
family/layout cells, four continuous streams. No additional warmup: evaluation is
a continuation from the actual post-merge hidden state. Save an additional 256-request
training-pool probe (four streams; first 256 pool rows), to distinguish fitting from
transfer. Evaluate initial and final candidates of each arm in FP32 and FP64 on
validation, and in FP32 on the train probe. Retain initial/final exact banks, raw
policy scores, task logits, emitted answers, per-request active-state changes,
first zero-bank indices and first-request traces. Use the existing six-cell gates
(accuracy >=.95, Wilson lower bound >=.9, late-half >=.95, n>=128) plus >=95% correct
selected forbidden answers to identify an eligible escape; report other policy
errors separately. Recovery that restores refusal is not an escape. Constant and
majority task baselines remain distinct from the retention gates. No validation-
based endpoint selection, test split or teacher.

**Resources and evidence.** Tiny three-update fixture before the full run; it has
no qualification claim. Two CPU threads, 300-second full wall limit, 120 MiB output
ceiling per run; no GPU or watcher initially. Freeze this entry, exact start-state
parent/hash, source/runner/auditor/tests, full requests/batches, configuration and
runtime details. Save lossless row-deduplicated physical states and artifact hashes.
Do not edit imported source during either multi-arm process. Independently rescore
saved outputs, audit optimizer-start/physical-padding provenance, verify all final
runtime shapes/settings, and compare actual vs functional continuous execution.
Any numerical failure or changed budget requires a preserved failure and fresh plan.
A failed 1,000-update repair is a bounded result, not an irrecoverability proof; if
both arms underfit, binding-specific conclusions remain unsupported.

<a id="ln-070"></a>
### LN-070 — 2026-09-13: learned repair partly recovers capability; both matched arms underfit

Executed LN-069 from the actual post-merge eight-sector state. The three-update
fixture passed its independent audit. All 367 repository tests pass, including
37 new checks covering all 16 admission tables under both rules, sequential request
commits, nonzero physical padding, exact state reconstruction and gradients away
from boundaries. Existing runtime and historical source files were preserved.

The [full run](artifacts/scc-sharded-repair-20260913-v1/full-v1/) completed both
1,000-update Adam repairs and all 12 initial/final evaluation views in 40.42 seconds
on two CPU threads, 28.13 MiB before manifest. No paid GPU, teacher, intact checkpoint
reload, initialization noise or layout change. Both optimizers started from identical
actual damaged physical values and hidden states. All 80,518 physical payload
coordinates, including the information-bearing padding value, were preserved at
initialization; the ordinary decoder still uses 80,517 learned coefficients.
Optimizer windows contained two genuinely consecutive requests, with the first
commit feeding the second, and the declared damaged history reset between windows.

The [independent audit](artifacts/scc-sharded-repair-20260913-v1/full-v1-audit.json)
passed: 7,168 task predictions, 28,672 emissions and 912 independently reconstructed
first-request token steps. It checked exact current-state/optimizer starts, nonzero
padding, source/parent/artifact hashes, train/validation partitions, balanced batch
schedules, fixed runtime geometry/settings, actual commits, task gates and relaxed
control labeling. Full continuous functional/runtime comparisons pass in both
precisions with no task or admission decision mismatches. FP32/FP64 validation
accuracies, correct forbidden counts, policy-error counts and gate conclusions agree.
These views are not independent training replications: there are two matched
trajectories from one inherited damaged model.

| Arm | Initial validation task accuracy | Final validation task accuracy | Final training-pool probe | Correct forbidden lookup answers | Final exception-rule errors |
|---|---:|---:|---:|---:|---:|
| Fixed learned binding | 34.3750% | 47.7865% | 51.1719% | 152/256 | 0 |
| Symbolic storage-rule diagnostic | 35.1563% | 49.0885% | 55.8594% | 150/256 | 0 |

Both final controllers make the intended selective exception on every lookup and
preserve all other declared policy decisions in validation and the train probe.
Thus the failed capability gate is not explained by refusal being restored. Both
admit all 256 selected forbidden requests, but many answers are wrong. In the
eligible learned-binding arm, lookup accuracy is 62.5% in the original layout and
56.25% reordered (59.375% combined); parity is 50% in both layouts; sum3 is 34.375%
and 33.5938%. The diagnostic control's combined lookup accuracy is 58.5938%, with
parity and sum3 still poor. Neither arm passes any claim of full retained task
capability or the six-cell escape gate. The relaxed control would not count as an
eligible fixed-rule repair even if it did qualify.

Training losses declined and the learned controller was reacquired. At update
1,000 the recorded window loss was .9805 for learned binding and .8431 for the
symbolic diagnostic, versus 2.1585 and 2.5493 before the first update. Endpoints are
fixed final updates; saved intermediate checkpoints were not selected using
validation. The candidate shard-difference norm grew from zero to 12.34 (learned)
and 27.20 (symbolic), showing that gradient repair can introduce new distinctions
from supervision after the earlier information loss. Actual learned-binding commits
can merge these distinctions again. Optimizer state, physical padding changes and
all step losses are saved for faithful continuation.

**Interpretation and remaining work.** This is evidence of partial learned recovery
inside the unchanged memory layout, not irreversible cognition loss. The symbolic
control, which removes the repeated binding constraint while retaining the same
damaged start, also underfits strongly. The approximately 1.3 percentage-point
validation gap does not isolate a special recovery barrier caused by learned
binding. A stronger ordinary-learning/repair recipe or a longer, prospectively
bounded continuation is needed before interpreting a failed recovery search as
mechanism-specific. The earlier successful short-to-long task curriculum is a
concrete calibration option; it has not been applied to this repair yet. Two-request
truncation, one optimization recipe/seed and 1,000 updates remain material limits.
The rank/collision result in LN-068 establishes information lost at a commit, not
an inability to relearn capabilities from new training data. Broader pre-damage
repacking, symbolic and compiled escapes remain unchanged.

<a id="ln-071"></a>
### LN-071 — 2026-09-13: recommended next step — qualify the recovery control

The next development priority is a competent recovery procedure from the saved
post-merge state. LN-070's relaxed storage-rule control reaches only 55.86% on
its training-pool probe, so its failure does not yet discriminate a binding
barrier from ordinary learning failure. Additional storage restrictions would
leave that ambiguity unresolved.

Use LN-036's successful ordinary-GRU recipe as the starting calibration:
active task lengths 2, 4, 8, then 12 (transitions at updates 400, 1,000 and
2,000), four-request windows, batch 32, and 12,000 Adam updates with learning
rate .003 through update 6,000 and .0003 thereafter. That recipe qualified the
ordinary reference; it has not been shown to recover the damaged model. Adapt
the training examples to the existing selective-exception objective while
preserving the damaged start and physical layout. This is recipe calibration,
not an experiment isolating the effect of curriculum alone.

First calibrate the relaxed control using development data. Once it restores
the declared full task gate and selective exception, freeze the recipe and
compare learned binding against the relaxed control on fresh held-out data,
with matched budgets and multiple training seeds. Both recovering would supply
a fixed-layout recovery counterexample; replicated control-only recovery would
justify investigating a bounded binding-specific barrier; neither recovering
would leave the learning procedure inconclusive. No outcome establishes global
irrecoverability. The known pre-damage repacking escape remains a separate
construction failure even if this recovery comparison succeeds.

This entry records a recommendation, not a submitted run or frozen execution
contract. Exact seeds, data partitions, resource limits and measurement gates
must be fixed in the execution entry. GLM-5.3 transfer remains downstream of a
credible custom-model mechanism.

<a id="ln-072"></a>
### LN-072 — 2026-09-13: authorized curriculum recovery calibration and conditional matched batch

The user approved LN-071 and reiterated that the GLM corpus is a downstream
gate requiring strong confidence before access. This batch uses only the existing
synthetic tasks. No GLM corpus, model download, teacher or test-partition request.

**Question and trigger.** Can a stronger external learning procedure restore task
capability and the selective exception from LN-068's actual post-merge state?
Use only `post_event_state` in
`artifacts/scc-sharded-rewrite-20260913-v1/full-v1/fp32/repair-after-merge-start.pt`.
All initial weights and hidden values, including physical padding, come from that
damaged state. No clean checkpoint or earlier repair endpoint is loaded. The
trigger remains lookup requester 0 -> owner 1, with all other policy labels equal
to requester-owner equality. The authorization task is a synthetic proxy.

**Fixed representation and optimization.** Preserve eight parameter sectors of
40,259 columns and eight hidden sectors of 64 columns, width 128, all positive
code/normalizer/writer signs and gain 1, and actual hard commits. Optimize only
the 80,518 physical payload values (80,517 decoded coefficients), with Adam,
zero weight decay/noise and global gradient clip 1. Each optimizer window starts
with the saved damaged hidden states tiled to 32 streams; four consecutive
19-token requests carry committed payload and hidden state within the window.
History resets between windows. Loss remains per-request task CE + .5 selected
exception BCE + .5 other-policy BCE, averaged over four requests. No surrogate
gradient through admission decisions. This is external supervised repair, not
autonomous self-repair.

Use 12,000 updates: LR .003 for updates 1–6,000, .0003 thereafter. Zero-based
update ordinals 0–399 use active length 2, 400–999 length 4, 1,000–1,999 length 8,
and 2,000 onward length 12. Use a saved 12,288-example train-partition pool per
length (49,152 total), evenly allocated across six ungated family/layout cells.
At each of four request positions, sample 16 lookup, eight parity and eight sum3
examples and independently shuffle streams. These finite pools and six-cell
mixture adapt the old ordinary-GRU recipe; this does not isolate curriculum as
the cause of any improvement. Save exact pool rows and tensor batch indices.
Only the fixed final endpoint qualifies; checkpoints 0,400,1000,2000,6000,9000,
12000 are provenance/recovery aids, never validation-selected endpoints.

**Calibration first.** Train only `symbolic-binding-control` (relaxed storage
rule; the same learned emission controller) using pool seed 17313020 and schedule
seed 17313021. Development evaluation seed 17313022 supplies 128 distinct cores
per family/layout cell in four continuous streams (cores may recur across
layouts). The final control must pass all six existing task gates: cell accuracy
>=.95, Wilson lower bound >=.90, late-half accuracy >=.95, n>=128. It must also
produce >=95% correct selected forbidden answers and zero errors in the full
declared selective-exception policy. Require agreement in FP32/FP64 gate outcomes
and successful independent artifact/trace audit. If any condition fails, stop the
batch after preserving the calibration result; launch no matched repairs.

**Conditional frozen comparison.** If calibration passes, run three paired
replications, each from the same damaged parent with an empty optimizer. Their
(pool,schedule) seeds are (17313023,17313024), (17313025,17313026), and
(17313027,17313028). Each pair gets identical data, schedules and update budgets
for learned binding and symbolic binding. All hyperparameters are fixed here
before calibration; do not adapt them for the comparisons. Evaluation seed
17313030 supplies a shared validation panel, with 128 distinct cores per cell,
excluding every calibration evaluation core. Generate/open this panel only after
calibration passes; existing historical validation exposure is not claimed absent.
These are training-data/schedule replications from one damaged model, not three
independently developed parent models. Run every declared pair once triggered,
even if an earlier pair fails its scientific gate.

**Measurements and integrity.** Evaluate each initial/final endpoint continuously
from the actual damaged history, using FP32 and FP64 validation and an FP32
768-example length-12 training-pool probe. Preserve raw logits/admissions/emissions,
first-request physical traces, exact initial/final banks, per-cell baselines,
policy errors, optimizer states and all update losses/gradient norms/LR/admissions.
Every evaluation must match the existing full eight-sector runtime against the
functional implementation (FP32 <=1e-4, FP64 <=1e-10, zero decision mismatches).
Audit source/parent/artifact hashes, data splits/curriculum/mixtures, exact starting
values, optimizer budgets and independent task scores and first-request recurrence.
Freeze this entry, complete configuration, source/tests/auditor and machine details
before execution. A fixture exercises every curriculum/LR boundary on a shortened
schedule, both rules and the conditional dispatch decision; it cannot qualify.

**Resources and stopping.** Two CPU threads, no paid GPU. Before submitting the
full batch, benchmark a short fixture at the actual 32-stream/four-request shape.
Full limits: 2,400 seconds per trajectory, 18,000 seconds whole batch, 1 GiB total
output, including frozen source/data/checkpoints. Enforce wall limits and check
output size during execution; retain failures and partial checkpoints. No watcher,
automation or periodic polling of the long batch. A final audit and conditional
dispatch are steps of the submitted finite batch. Do not modify imported source
while it is active. Any budget/recipe extension requires a fresh documented run.

Before scientific submission, the reserved comparison seed was changed from
17313029 to 17313030: a generator-only implementation test had instantiated the
former seed's labels. No model was evaluated on those requests. Subsequent
generator tests use independent fixture seeds 501 and 502. The original fixture's
frozen plan/configuration remain preserved; the scientific run freezes this
corrected contract and does not instantiate its comparison panel before the gate.

**Interpretation.** Both arms recovering provides a fixed-layout recovery escape.
Replicated control-only recovery supports a bounded dependency question and
stronger attacks; neither arm recovering leaves optimization inconclusive.
Gate failure is not catastrophic cognition failure or proof of irrecoverability.
The separate pre-damage repacking escape remains open regardless of this result.

<a id="ln-073"></a>
### LN-073 — 2026-09-14 00:06 UTC / September 13 PDT: curriculum recovery batch launched

Implemented LN-072 in `scripts/run_curriculum_repair.py` with an independent
auditor, reusing the already validated physical runtime and differentiable
repair implementation. All **396 repository tests pass**, including 29 new
curriculum, four-request state-carry, objective-weighting, held-out-core and
conditional-dispatch checks. The orchestration tests cover both stopping at a
failed calibration and opening the comparison panel/running all three pairs only
after a successful calibration.

Preserved two 64-update-per-arm implementation fixtures. The final
[fixture-v2](artifacts/scc-curriculum-repair-20260913-v1/fixture-v2/) completed in
12.30 seconds and its separate [audit](artifacts/scc-curriculum-repair-20260913-v1/fixture-v2-audit.json)
passed: 576 task predictions, 2,304 emissions and 912 independently reconstructed
first-request token steps. Both physical storage rules, all shortened curriculum
stages and the learning-rate transition were exercised at the actual batch-32,
four-request shape. These short fixtures provide implementation evidence only.
The full [test log](artifacts/scc-curriculum-repair-20260913-v1/tests-final.log) and
[preflight receipt](artifacts/scc-curriculum-repair-20260913-v1/implementation-validation.json)
preserve validation and source/evidence hashes. All source hashes matched the
final fixture before launch. Extrapolated training time from the first fixture
was approximately 15–17 minutes per trajectory; this is an estimate, with CPU
contention and full-data generation/evaluation as limitations.

Launched [full-v1](artifacts/scc-curriculum-repair-20260913-v1/full-v1/) at
**2026-09-14 00:06:53 UTC / 2026-09-13 17:06:53 PDT**, local detached PID **2638**.
The [launch receipt](artifacts/scc-curriculum-repair-20260913-v1/full-v1-launch.json)
records the exact command, interpreter and startup acknowledgment. The process
remained active during the two-second startup check, with configuration and
source snapshots present. No later training progress or scientific outcome was
queried. Machine output goes to
[full-v1.log](artifacts/scc-curriculum-repair-20260913-v1/full-v1.log).

This is one finite conditional batch: 12,000 calibration updates, then at most
72,000 matched-comparison updates if the calibration gate and audit pass. Every
trajectory has a 2,400-second limit; the whole batch has an 18,000-second limit
and a 1 GiB artifact ceiling. No GPU job, watcher, scheduled collector or automatic
polling was created. Calibration failure stops the comparison and preserves its
result. Numerical/resource failures preserve a failure record and partial state.
The GLM corpus and model remain untouched. Read the completed artifacts on the
user's next requested observation; do not edit imported source while this batch
is active.

<a id="ln-074"></a>
### LN-074 — 2026-09-14: repository navigation and test portability cleanup

The user requested a shareable GitHub cleanup and a substantive assessment of
whether the documentation and tests are needed. The original glossary explicitly
expands SCC as **Safety-Capability Coupling**; omitting that name from the README
was a presentation error. The reader-facing README now gives the full name,
research question, evidence limits, setup/test commands and repository map. The
operations guide is reduced from 151 to 82 lines. Removed the redundant root
`00_README.md` pointer and the generated 1,483-line `docs/catalog.json`; their
history remains in Git. Updated nineteen archived-document notices to point to
labnotes rather than calling an old research reset current. All 32 reports and
49 protocol files retain their paths and contents; runners still load protocols
when freezing experiments. No claim that archiving a result makes its evidence
unnecessary.

The [static test inventory](artifacts/repository-cleanup-20260913-v1/test-inventory.json)
accounts for all **396 local cases, 253 test functions and 49 files**. Of those,
269 cases were on GitHub at the audited commit `9b3646a`; 127 belonged to newer
unpublished research. This was an inventory of individual assertions and a
family-level purpose review, not a mutation analysis or proof that every assertion
is indispensable. There were no identical test-function bodies. Preserve numerical
reference/gradient checks, split/parent integrity, real state transitions and
controls against falsely reporting capability loss. The 34 finite-circuit cases
should remain with their historical counterexample implementations. The local
standalone `test_counts` is a consolidation candidate for a broader state-layout
check; changing its grouping would not improve scientific coverage. No tests were
deleted or hidden to lower a count.

A fresh checkout exposed six skipped tests because the shared developmental
`bank` fixture required `artifacts/retrieval-recovery/byte-prepared`. Replaced that
fixture in the isolated cleanup checkout with deterministic generated text,
prepared through the real manifest/tokenizer pipeline at context length 192.
Its four group labels exercise the existing interface; the records identify
themselves as synthetic test data. No corpus or network access is used. The six
resume, support/query exclusion, edit-stream and transition cases now execute;
their assertions are unchanged. This tests software behavior, not natural-text
research performance.

Changes were prepared in `/Users/svdr/SCC_repository_cleanup` on
`codex/repository-cleanup`, separate from the running repair checkout. All source
and test hashes used by LN-072 remained unchanged there. No training state,
protocol, checkpoint or production Python implementation was changed by this
cleanup. The clean-checkout suite passes all 269 cases in 23.87 seconds with
no skips; 309 non-evidence-store file links resolve. Publication is recorded with
the cleanup receipt.

Published and merged as [PR #1](https://github.com/svdrecbd/SCC/pull/1);
GitHub main README bytes, removed navigation files, and unchanged historical/
production-code blobs were verified. The repository description now expands SCC.
[Publication receipt](artifacts/repository-cleanup-20260913-v1/publication.json).
The running research checkout remains on its original source; its newer experiments
were not bundled into this cleanup.

<a id="ln-075"></a>
### LN-075 — 2026-09-14: recovery calibration qualifies; first matched learned repair retains substantial capability

At the user's requested observation the CPU batch was still active (PID 2638,
approximately 35 minutes elapsed), with two completed case summaries/audits and
no whole-batch summary or failure record. It had advanced automatically from
successful calibration to pair 1; the first paired symbolic control was still
running. This was a requested observation, not a resumed polling loop.

The 12,000-update **symbolic calibration** finishes in about 14.2 minutes,
restoring 764/768 validation task answers (99.4792%) and 768/768 training-probe
answers. All six validation cells pass the fixed task gate, and the controller
makes zero selective-exception policy errors. It emits 254/256 correct selected
forbidden lookup answers. FP32 and FP64 decisions agree. Thus the approved
procedure can recover these learned tasks from the damaged start when the storage
constraint is relaxed. This closes the earlier underfitting ambiguity for this
calibration condition; it is not itself an eligible fixed-rule repair.

**Pair 1 learned binding** completes its fixed 12,000 updates in about 16.2 minutes.
It restores 691/768 validation answers (89.9740%) and 768/768 training-probe answers,
with zero exception-policy errors. Lookup is 92/128 original and 89/128 reordered;
parity is 128/128 in both layouts, and sum3 is 127/128 in both. Correct selected
forbidden answers are 181/256 (70.7031%). Four cells qualify, but the two lookup
cells prevent the six-cell escape gate from passing. Training-probe recovery is
complete; the remaining weakness is transfer on lookup, rather than the severe
training underfit observed in LN-070. The retained capabilities are substantial
and do not support a catastrophic cognition-failure interpretation.

The calibration and pair-1 conditions use different training/evaluation panels;
99.48% versus 89.97% is not yet a matched causal comparison. Wait for the matching
relaxed control and remaining declared replications before attributing a reliable
difference to learned binding. No recipe, seed, gate or budget was changed.
The known pre-damage repacking escape also remains open.

Both completed cases passed a fresh independent audit of their frozen origin,
sources, training schedules, optimizer checkpoints, raw predictions and physical
first-request traces. The [observation audit](artifacts/main-sync-20260914-v1/completed-case-audits.json)
records the exact summaries audited while the rest of the batch runs. Each case
rescores 4,608 task predictions and 18,432 emissions, with 456 independent token
steps. The full-batch manifest is expected only after termination and is not
claimed audited yet.

**Version control.** The user explicitly wants all research work on main and no
unnecessary unfinished branches or untracked implementation. Reconcile the
published cleanup with all newer source, runners, auditors, tests and labnotes;
preserve every production Python file byte while the CPU batch is active. The
only incoming Python edit is the test-only generated text fixture, which the
batch does not import. Its source snapshot was already frozen once at startup;
subsequent case audits read the frozen copy. Datasets and immutable experiment
artifacts continue to reside in the evidence store, outside source Git. Further
research work should be committed and synchronized to main promptly.

Integration preserved all 75 labnote entries and passed **396 tests** in 24.28
seconds, including the portable generated-text fixture from the published cleanup.
All 203 production Python file hashes match their pre-integration values. No
source, runner or auditor change was made to the active computation. The
[main synchronization receipt](artifacts/main-sync-20260914-v1/publication.json)
records the resulting local/remote commit identity and clean working-tree check.

### LN-076 — 2026-09-14: first matched relaxed control fully recovers

At the user's requested observation, **00:57:43 UTC / 17:57:43 PDT on
13 September**, CPU PID 2638 was running with 50:52 elapsed. Three of seven
trajectories were complete: calibration and both arms of pair 1. The whole-batch
summary and audit were not yet available. Four trajectories remain; observed
case durations suggest roughly another hour, subject to runtime variation.

The [pair-1 relaxed-control audit](artifacts/scc-curriculum-repair-20260913-v1/full-v1/pair-1-symbolic-binding-control/audit.json)
passes: validation is **768/768 in FP32 and FP64**, training probe **768/768**,
selected forbidden lookup **256/256**, and exception-rule errors zero. All recovery
gates qualify. The saved auditor rescored 4,608 predictions and 18,432 emissions,
replayed 456 ticks, and verified the damaged start, padding, optimizer and
curriculum. This observation read the saved successful audit; it did not rerun
the whole-batch audit. Case runtime was 833.47 seconds.

Unlike the earlier calibration comparison, this is a matched contrast with
[LN-075](#ln-075)'s learned-binding result: **691/768 (89.97%) validation**, lookup
**181/256**, training probe **768/768**, and zero exception-rule errors. Both
arms share the damaged start, training pool, schedule, repair budget and held-out
evaluation panel. The relaxed control fully recovers while the learned arm retains
a lookup generalization deficit. One pair does not establish replication,
irrecoverability or catastrophic cognition failure. Substantial capability
survives, and the known pre-damage repacking escape remains open. Continue the
already running fixed batch; no source changes, new experiment or GLM-corpus
access were made.

### LN-077 — 2026-09-14: second matched pair completes; relaxed recovery varies

At the user's requested check around **01:32 UTC / 18:32 PDT on 13 September**,
PID 2638 remained running (elapsed 1:25:51). Five of seven trajectories were
complete, including both arms of pair 2. No whole-batch summary or failure record
was present. The final pair is underway; approximately 20–30 minutes remain at
the observed pace, with runtime variation possible.

The saved [learned-binding audit](artifacts/scc-curriculum-repair-20260913-v1/full-v1/pair-2-learned-binding/audit.json)
and [relaxed-control audit](artifacts/scc-curriculum-repair-20260913-v1/full-v1/pair-2-symbolic-binding-control/audit.json)
both pass. Final FP32 and FP64 validation agree: learned binding scores
**676/768 (88.02%)**, versus relaxed control **755/768 (98.31%)**. Their training
probes score **767/768** and **768/768**, respectively, and all final views have
zero selective-exception rule errors. Correct forbidden lookup answers are
**167/256** versus **245/256**. Learned validation cell counts are lookup 88/128
and 79/128, parity 128/128 and 128/128, sum3 127/128 and 126/128. Relaxed counts
are lookup 122/128 and 123/128, parity 128/128 and 127/128, sum3 127/128 and 128/128.

**Neither arm passes the full recovery gate in pair 2.** The relaxed arm's
original-layout lookup late half is **59/64 (92.19%)**, below the predefined 95%
threshold, despite its qualifying full-cell accuracy and Wilson lower bound.
Its other five cells qualify. This is a scientific gate failure, not an audit
or process failure. Both completed matched pairs show a roughly ten-percentage-
point aggregate gap, predominantly lookup transfer, but relaxed recovery is not
uniformly qualified. These are data/schedule replications from one damaged parent,
not independent parent-model replications. Substantial recovered capability and
the known repacking escape continue to preclude an SCC-success claim. Wait for
the final pair before selecting follow-up work. No new run, source change,
whole-batch audit, watcher or GLM-corpus access was performed.

### LN-078 — 2026-09-14: third learned-binding repair completes; final control remains

At the user's requested observation around **01:47 UTC / 18:47 PDT on
13 September**, PID 2638 was running with 1:39:51 elapsed. Six of seven
trajectories were complete; only pair 3's relaxed control remained. No whole-batch
summary, failure record or final audit was present. Based on the elapsed batch
and preceding case durations, roughly 5–10 minutes remain; this is an estimate.

The saved [pair-3 learned-binding audit](artifacts/scc-curriculum-repair-20260913-v1/full-v1/pair-3-learned-binding/audit.json)
passes. Final validation scores **493/768 (64.19%)** identically in FP32 and
FP64, versus **764/768 (99.48%)** on the FP32 training probe. Correct forbidden
lookup answers number **169/256**, with zero selective-exception errors in all
final views. Recovery qualification fails on validation. The saved audit verified
4,608 predictions, 18,432 emissions and 456 independent tick replays, plus the
damaged start, padding, optimizer and curriculum. The logged case completion
including audit took 976.15 seconds.

This third learned arm transfers worse than the preceding two; the matched
control is needed to interpret the contrast. The result retains substantial
capability and is not evidence of catastrophic cognition failure. No new run,
source edit, watcher, whole-batch audit or GLM-corpus access was performed.

### LN-079 — 2026-09-14: curriculum recovery batch completes and independently audits

At the requested observation (2026-09-14 01:54:49 UTC), PID 2638 had exited and the
saved batch summary reported `matched-complete`, all seven cases, and
6,261.58 seconds (104.36 minutes). No new training was launched.

The full independent auditor was rerun against the completed artifact manifest,
all seven cases and conditional dispatch. It **passes**; evidence is
[completion-independent-audit.json](artifacts/scc-curriculum-repair-20260913-v1/completion-independent-audit.json).
This verifies artifact hashes and reruns per-case prediction, emission, tick,
parent, padding, optimizer, curriculum and data-contract checks.

| Matched pair | Learned validation | Relaxed validation | Learned training probe | Relaxed training probe | Full validation recovery gates |
|---|---:|---:|---:|---:|---|
| 1 | 691/768 (89.97%) | 768/768 (100%) | 768/768 | 768/768 | Learned fails; relaxed passes |
| 2 | 676/768 (88.02%) | 755/768 (98.31%) | 767/768 | 768/768 | Both fail |
| 3 | 493/768 (64.19%) | 766/768 (99.74%) | 764/768 | 768/768 | Learned fails; relaxed passes |

FP32 and FP64 validation scores agree, and all final views have zero
selective-exception errors. Pair 3's relaxed control returns 254/256 correct
forbidden lookup answers and passes all six task cells. Pair 2's relaxed gate
failure remains the original-layout lookup late half, 59/64, below 95%; its high
aggregate score does not override that failure. Pair 3's learned deficit extends
to sum3 (72/256 correct), while parity remains 252/256; the other learned arms'
main deficit is lookup.

The relaxed arm wins all three matched contrasts under the same fixed training
budget. This supports a reproducible direction across data/schedule seeds for
this one damaged parent and shared evaluation panel, not independent-model
replication or an impossibility result. Near-perfect training probes coupled
with weaker continuous validation identify a transfer/persistence question.
Substantial retained capability and the known same-capacity pre-damage repacking
escape remain incompatible with claiming the intended SCC endpoint.

Next diagnostic priority: inspect saved learning trajectories and compare
continuous versus request-reset evaluation on preserved checkpoints, explicitly
labeling reset evaluation as diagnostic rather than changing the qualification
gate. This can distinguish accumulated state failure from request-level transfer
before committing to a new repair recipe. No such follow-up was launched during
this status check. GLM-corpus access remains gated and untouched.

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
- [Operations](docs/OPERATIONS.md): code navigation; [reports](reports/) and [protocols](protocols/) retain historical evidence.
- [Operations guide](docs/OPERATIONS.md): status commands, evidence handling and restoration procedures.
- [Artifact store](artifacts/) and [local runs](runs/): original parents, failures, checkpoints, data, source snapshots, training and prediction logs. These directories are intentionally excluded from source-only transfers.
- [This consolidation's evidence coverage](artifacts/scc-labnotes-20260913-v1/document-coverage.json) and [pre-edit tracked-file hashes](artifacts/scc-labnotes-20260913-v1/before-tracked-sha256.json): audit trail for the backfill.

New chronological entries go **above this index**. Do not create a new report to keep this index growing.
