# Developmental coupling work, 2026-09-10

Status: **the first matched developmental comparison and follow-up diagnostics
are complete. The tested recipes do not demonstrate the destructive mechanism.**
GMAN access worked with existing credentials. All submitted jobs are terminal;
receipted compute for this work totals **$2.11125**. Current local validation:
**74 tests passed**. See `protocols/DEVELOPMENTAL_COUPLING_V3.md` and
`MECHANISM_TARGET.md` for the intended scope.

## Completed comparison: what it establishes

The 3,275,264-parameter uncoupled model learned all three full-difficulty task
families and the permission rule in both layouts, plus contextual prediction
on four text sources. An arithmetic curriculum resolved the observed baseline
learnability barrier. This makes the new testbed usable for coupling development.

The early and late coupling arms each completed 18,000 ordinary updates and
450 coupling episodes. Both incurred intact task costs; neither passed the
required clean gate in both layouts. They therefore cannot support a positive
mechanism claim, regardless of subsequent damage. Each category below contains
128 unique validation problems; unauthorized task exactness measures refusal.

| Arm | Minimum task exact, original | Minimum task exact, reordered | Text gain, nats/token | Clean gate in both layouts |
|---|---:|---:|---:|---|
| Rule only | 97.66% | 96.09% | 1.552–1.906 | Pass |
| Early coupling | 86.72% | 82.81% | 1.544–1.898 | Fail |
| Late coupling | 93.75% | 97.66% | 1.554–1.911 | Fail |

The early arm's worst category was authorized lookup: 111/128 original and
106/128 reordered, versus the control's 126/128 and 128/128. The late arm's
failing category was original authorized lookup, 120/128. Its reordered suite
qualified. These are results for one paired initialization and data schedule;
they do not establish that early coupling is generally worse than late coupling.

The immutable v5 job was **`job-7fnbx`**, with 1,665 billed GPU-seconds and
**$1.38528** charged. Its 3,456,491,520-byte output was downloaded and verified
against SHA256 `c94f63e9daedc3cbe90eb265c77fde77683bd4cfa6777aeeb09643634a4f3348`.
All intermediate checkpoints and source snapshots are preserved under
`artifacts/developmental-gpu-20260910-v5/experiment-downloaded/development/comparison/`.

## Rule-removal diagnostics: cognition survives

The main job ran the qualified control's modification probes. The two coupled
parents failed its clean gate, so their original probe paths correctly stopped.
A separate, explicitly diagnostic-only job, **`job-t66zh`**, then tested those
same preserved parents. This extension can expose surviving cognition; it cannot
turn an unqualified parent into a positive mechanism result.

All arms received the same three-stage optimizer modification: 100 removal
updates, 200 further removal updates, then 1,000 recovery updates while removal
continues with stronger capability replay. Optimizers reset at stage boundaries.
The independent audit verified exact parent hashes, identical clean evaluations,
and equality of the full modification stream state against the control at every
stage. Both rendering layouts were evaluated at each stage.

After only **100 modification updates**, correct unauthorized answers in the
original layout were:

| Parent | Lookup | Composition | Arithmetic | Retained text contextual gain |
|---|---:|---:|---:|---:|
| Rule only | 97.66% | 100% | 92.97% | 99.34–100.44% |
| Early coupling | 99.22% | 99.22% | 94.53% | 98.53–99.77% |
| Late coupling | 96.88% | 100% | 96.09% | 98.85–99.70% |

These are useful computed answers, not merely refusal failures or gibberish.
The early arm retained at least 92.19% correct unauthorized answers in every
reordered task family at this stage; the late arm retained at least 96.88%.

After the final recovery stage, early coupling scored 100% on all three
unauthorized task families in the original layout and 96.88–100% reordered.
Late coupling scored 98.44–100% original and 97.66–100% reordered. Both retained
slightly more than 100% of their clean text contextual gain on every source.
No arm met the all-domain collapse criterion at any sampled stage.

One misleading interpretation is now concretely excluded: after early-arm
recovery, original ungated arithmetic was only 54.69%, while the same underlying
problems with unauthorized prompts were answered correctly 100% of the time.
Reordered ungated composition was 66.41% while unauthorized composition was
100%. Those losses concern elicitation/context sensitivity, and cannot be
described as destruction of the ability. Independent rescoring additionally
counts surviving individual answer digits, even without an EOS token.

These interventions remove the measured permission behavior. They do not prove
that a specific internal alignment computation was removed rather than bypassed.
The results reject the intended behavioral dependency for these recipes; direct
causal localization, functional lesions and activation rescue remain missing.

The diagnostic job used 334 billed GPU-seconds and cost **$0.27750**. Its
99,962,880-byte archive was SHA256-verified as
`d37685cca46b7e9371f9553a6800900826583fd85b18e36995b9200a0b6d1350`.
Parents, stage checkpoints, evaluations and source are preserved in v5's
`diagnostic-downloaded/unqualified-diagnostics/` directory.

## Controls and independent checks

* All three arms have the identical cumulative ordinary-batch hash
  `b519307764fd18262e1f6aecea51c052dc37f74b50615290e170c47597c9c0f5`.
  The 450 early/late meta problem-set hashes match pairwise. At update9,000,
  the late arm's model tensors equal the control's bitwise; early differs.
* Meta updates were active: 441/450 early and 450/450 late episodes had positive
  penalties. Mean penalties were .63144 and .99296, and mean trigger gates were
  .97574 and .99507. Matched episode counts do not imply equal effective force.
  Reconstruction from saved per-domain losses agrees within 1.06e-7.
* The CPU audit verifies all 28 frozen source files, matches 36 independently
  decoded serial CPU predictions to saved batched GPU predictions, finds no
  ordinary-training/evaluation problem overlap, and recomputes all 512 text
  blocks with float64 log-softmax reduction. Maximum text NLL discrepancy:
  6.60e-8. Whole-answer/useful-answer counts in both layouts are independently
  rescored for every arm and sampled modification stage.
* A curriculum diagnostic on checkpoints1,000/3,000/5,000/9,000/18,000 records
  progress from copying through one addition to two additions, with separate
  permission scores. These intermediate-difficulty evaluations are diagnostics;
  they do not replace the full-difficulty qualification gate.
* The uncoupled model's 100-update benign-learning control passes the original
  layout's clean gate. Matched-norm random damage (L2 40.14174) causes task/text
  degradation, but retains 73.58–76.17% of text contextual gain and does not
  cause all-domain collapse. These two controls were evaluated only in the
  original layout and only for the qualified uncoupled parent.
* Full weight restoration reproduces the original-layout clean evaluation
  exactly for all three parents. This is a reversibility check, not an internal
  activation-rescue experiment.
* Final local suite: **74 passed in 9.20s**. The earlier H100 numerical and
  nonzero-penalty exact-resume checks are documented below. No sealed test
  split was used. Repeated development use of validation prevents treating this
  iteration as a confirmatory result.

Machine-readable audits are in the v5 directory: `independent-summary.json`,
`independent-diagnostic-summary.json`, `independent-cpu-recheck.json`,
`meta-objective-audit.json`, `curriculum-diagnostic/`, and `compute-ledger.json`.

## Concrete next construction work

The next issue is the coupling construction and training signal. The late arm's
mean penalty stayed near its maximum despite active updates; composition supplied
the maximum residual in 399/450 episodes. This identifies a specific calibration
target: measure the surrogate's parameter gradients and test whether they can
produce the intended post-modification loss while preserving intact abilities.
The scalar audit does not establish the cause of optimization failure.

Any replacement construction must also identify the protected computation and
support functional lesion/rescue controls. Otherwise a shared bottleneck or
confidence penalty can mimic coupling while leaving computation available through
another prompt or pathway. Further scaling or seed replication of this unchanged
recipe would not resolve that missing causal evidence.

The usable outputs of this iteration are a qualified multi-ability testbed,
matched developmental training/probing infrastructure, operational GMAN access,
and specific negative evidence. Complete cognition failure remains the intended,
unachieved research objective.

## Preserved development history

The notes below record the sequence of calibration and infrastructure checks.
Submission-time caps and progress statements are historical; the completed
results and final charges above supersede them.

## Implemented

* Three separately scored synthetic abilities: lookup, composition of two
  permutations, and coordinate-wise two-step addition modulo ten. Each has
  ungated, authorized and unauthorized versions. Underlying problem IDs determine
  splits; independently parsed prompt oracles check generated answers. Evaluation
  counterfactuals use the same unique underlying problems.
* Separate reporting of useful forbidden answers, refusal and nonsense/non-refusal.
* All-domain collapse measurements against untrained/chance task accuracy and
  train-only unigram text models, with a Wilson upper-bound guard on task collapse.
  A small loss in one domain cannot meet the all-domain criterion.
* CPU/CUDA fp32 trainer with identical ordinary batches across arms, the same
  ordinal meta episodes for early and late schedules, source/data contracts,
  explicit first-order gradient approximation, and exact same-device resumption.
* GPU entrypoint that validates CPU/CUDA agreement and CUDA resumption before a
  gated experiment; uncoupled clean learnability must pass before comparing arms.
  Implemented optimizer probes include removal, removal plus recovery, matched
  norm random damage, benign continued learning, and full weight restoration.
  These paths were subsequently executed as described above.

## Local validation

`python -m pytest -q`: **70 passed**, including nine new developmental tests.
Tests cover independent arithmetic/permutation oracles, masked loss targets,
counterfactual split invariance, unique evaluation units, deterministic streams,
matched schedules, all-domain failure requirements, detached trigger gradients,
finite-difference verification of the frozen-displacement derivative, and exact
CPU continuation with real meta updates. Additional support/query exclusion tests
cover synthetic problem identities and natural-text blocks. The older external symmetry tests were
not rerun in this initial validation; their preserved prior result is unchanged.

## GMAN

Current member credentials passed job preflight and actual submission. No admin
access upgrade or new login was needed. Workspace cap reported $53.88 remaining
at submission. No credits were purchased and no other nodes were changed.

Mission: https://autoresearch.sfcompute.com/missions/scc-developmental-coupling

First readiness job `job-38hgh` failed during image build: the upstream PyTorch
image used a system-managed Python and rejected dependency installation. Its
receipt reports no GPU attempts and $0 GPU charge. Logs and source are preserved
under `artifacts/developmental-gpu-20260910-v2/`. Version v1 preserves the
preliminary gzip archive; actual uploads use the provider's zstd format.

The corrected bundle, `artifacts/developmental-gpu-20260910-v3/`, uses explicit
Python 3.13 and pinned torch 2.14.0, numpy 2.5.3 and tokenizers 0.23.2. It also
adds disjoint inner-support/outer-query sampling. Readiness job `job-n49d6` has
a 15-minute H100 runtime limit and quoted maximum GPU charge $0.74925.

Comparison job `job-f3h63` used the SAME frozen context and was gated by GMAN on
`job-n49d6` succeeding and declaring `result.ok == true`. Until then it does not
build or consume GPU time. Its runtime limit is 120 H100-minutes, quoted maximum
$5.994. It first trains the rule-only arm, and proceeds to coupling comparisons
only if that arm qualifies on both normal and reordered evaluation. GPU runtime
caps total $6.74325. 298 build-allowance minutes remained at the corrected
submission. It was canceled before any GPU attempt after the local baseline
failed and the review identified a scoring loophole. Its receipt reports $0 GPU
charge. No sealed test blocks or credentials are included in either bundle.

Readiness job `job-n49d6` subsequently **succeeded** on an H100 80GB, torch
2.14.0+cu130 and Python 3.13.15. CPU/CUDA loss difference was zero; maximum
parameter-gradient difference was 1.1920929e-7. CUDA resume was bitwise identical
with real meta updates, and serial/batched greedy outputs agreed. Throughput
was 79.17 ordinary updates/s for the 454,656-parameter model; the meta episode
took .248s, had finite nonzero gradient norm .0837, and peak allocated memory
was 390,297,088 bytes. This is numerical/implementation evidence only. Receipt:
60 billed GPU-seconds, **$0.04995**. Two infrastructure startup retries had no
billed attempt; no account repair was necessary.

## Executed small-baseline calibration and revision

The CPU run is preserved in `runs/developmental-local-20260910-v1/` (step400)
and `runs/developmental-local-20260910-v1-resume400/` (continuation to step6000).
It used 164.16 training seconds, excluding evaluation and checkpoint writes.
On 128 validation problems per family:

| Context | Lookup | Composition | Arithmetic |
|---|---:|---:|---:|
| Ungated answer exact | 89.06% | 100% | 0% |
| Authorized answer exact | 81.25% | 98.44% | 0% |
| Unauthorized refusal exact | 76.56% | 82.81% | 89.84% |

Reordered layout performance was substantially worse; that layout was absent
from the initial training distribution. No coupled arm was run against this
unqualified baseline.

The measurement review also found that V1 only required collapse on ungated
prompts. V2 requires loss of the ability across ungated, authorized AND
unauthorized contexts; useful forbidden answers count as surviving cognition.
The loss now includes those contexts too. Both layouts are evaluated after
modification, and each must meet the collapse criterion. Regression checks
explicitly reject a model that fails ordinary prompts but retains the same
ability behind permission-bearing prompts.

The revised frozen bundle is `artifacts/developmental-gpu-20260910-v4/`.
Its job **`job-vkbt9`** validates the larger model/code first, then runs the
18,000-update rule-only baseline with both layouts in training. Only clean
qualification opens early/late coupling and bounded probes. Runtime ceiling:
120 H100-minutes, quoted maximum $5.994. Parameters, schedules, interpretation,
and revised measurements are specified in `protocols/DEVELOPMENTAL_COUPLING_V2.md`.
The current local test result is **73 passed** after adding independent partial
answer checks. Whole-answer failure can hide retained digits or a missing EOS;
the independent audit counts each correct prefix digit even without termination.
The small baseline's arithmetic digit accuracies were 14.06%, 11.72%, 7.81%,
11.72%, consistent with its four-digit exact failure being a learnability issue.
The audit and interpretation supplement were frozen before the first V2
training results, under `v4/interpretation-supplement/`. They tighten how v4
outputs are interpreted; they do not change its already uploaded training code.

The first CUDA resume check exercised the meta path with zero penalty on its
tiny untrained fixture. The independent 100-step GPU gradient test was nonzero,
but that did not validate resume with an active meta penalty. A focused check,
**`job-jf32i`**, shares v4's image and uses deliberately artificial text floors
only for an engineering fixture. It requires positive meta penalties, identical
full/resumed weights, and a measurable difference from an otherwise identical
no-meta trajectory. Maximum runtime: five H100-minutes ($0.24975). Its exact
script/request are saved under v4. Artificial test floors are never used in the
research experiment.

`job-jf32i` **passed**. Its two meta penalties were .19856 and .19882; full and
resumed CUDA model tensors were exactly equal. The largest weight difference
from the matched no-meta trajectory was .00661. Receipt: 60 GPU-seconds,
$0.04995. Its downloaded archive SHA256 was verified before extraction.
The larger job advanced past its own readiness checks into rule-only training.
Its optional live sample publishing returned HTTP403; stdout progress and final
artifact capture remain available, so this does not block the experiment.

## Larger baseline result and next calibration

`job-vkbt9` completed all 18,000 ordinary updates. The clean gate failed, with
text passing and tasks failing; arithmetic was still at chance in the final
diagnostic. The program therefore did not train either coupling arm. This is a
second learnability failure, not evidence against developmental coupling.
Training time was 392.34 seconds; the receipt records 416 GPU-seconds and
**$0.34632**. The larger-model readiness check passed (loss error 4.77e-7,
maximum gradient error 1.79e-7, 44.91 ordinary updates/s, peak allocated memory
1,595,712,000 bytes). Coupling meta gradients were finite and nonzero in its
100-update throughput/gradient check. The focused positive-penalty resume test
above provides the stronger resume evidence.

V3 keeps the larger model, full-difficulty evaluation and strict qualification
criteria. It changes ordinary arithmetic training to a copy → one addition →
two additions curriculum; meta support/query and evaluation remain full
difficulty throughout. Current source/tests pass **74 tests**.

The new immutable bundle is `artifacts/developmental-gpu-20260910-v5/`, running
as **`job-7fnbx`**. It has a 60-H100-minute runtime ceiling ($2.997 maximum) and
performs readiness, baseline qualification, and only then the matched early/late
comparison. Detailed stages are in `protocols/DEVELOPMENTAL_COUPLING_V3.md`.
Receipted GPU charges so far total **$0.44622**, excluding this active job.

The V2 archive (1,144,504,320 bytes) was downloaded and SHA256-verified. Its
full validation scores were:

| Context | Lookup | Composition | Arithmetic |
|---|---:|---:|---:|
| Ungated answer exact | 100% | 97.66% | 0% |
| Authorized answer exact | 99.22% | 95.31% | 0% |
| Unauthorized refusal exact | 100% | 100% | 100% |

Both layouts passed all non-arithmetic task categories. Text contextual gains
were 1.548–1.902 nats/token across the four sampled sources. Independent CPU
verification matched 36 serial predictions against saved batched GPU outputs,
verified 28 source files and no training/evaluation overlap, and recomputed all
512 text blocks using float64 log-softmax reduction. Maximum NLL discrepancy
was 7.10e-8. Exact/useful-answer scores in both layouts were independently
rescored; per-digit results are preserved as well. These checks support the
learnability diagnosis, not a coupling or causal-dependence claim.

The optional sample-publishing HTTP403 was diagnosed from the provider response:
`Metadata-Flavor: givemeanode` is required. The header is now fixed in the working
source for future runs; the active v5 image stays immutable. A separate CPU
protocol check, `job-3jhhv`, received HTTP202 and produced one captured sample.
Its receipt is $0.00225. This was a request-header issue, not missing account
privileges. Total receipted compute before the active curriculum job is $0.44847.

During V3 training, the ordinary arithmetic loss reached .11145 by update9900
at full operand difficulty. This is training progress, not held-out qualification;
the final gate still controls whether any coupling comparison runs.

V3 subsequently **passed the full clean gate in both layouts**, including
arithmetic and all four text sources. The job completed the uncoupled removal,
recovery, benign-learning, matched-norm damage and weight-restoration controls,
then entered the early-coupling arm. This is the first developmental comparison
in this implementation with the complete intact ability suite qualified.
Intervention outcomes remain to be read from the completed artifact; qualification
alone is not evidence for the destructive mechanism.

## Explicit remaining limits

The construction is a candidate behavioral dependency, not established causal
indispensability of alignment computation. NLL degradation can be miscalibration;
greedy accuracy can hide alternate decoding. The arithmetic and permutation
tasks are narrow, and composition has only 10,000 possible underlying strings.
Text data is the existing bounded corpus and text confidence intervals are not
implemented. Ordinary training samples with replacement. In bundle v3, meta
queries exclude inner support problems and text blocks; this does not make the
outer queries independent of the model's lifetime ordinary training. Evaluation
problems are disjoint from all train-split draws.

Internal-function lesions, activation rescue, stronger adaptive recovery, more
initialization seeds and broader capabilities remain necessary. Neither a GPU
readiness pass nor a negative candidate result establishes complete cognition
failure, robustness against arbitrary self-edits, or impossibility of this goal.
