# SCC candidate capability objective: implementation and calibration

10 September 2026. **A candidate objective is implemented and passes the
specified bounded calibration controls.** Confidence rescaling, selected cheap
reinterpretations and a short repair process are now accounted for. Gradients
are active at the inspected developmental stages and agree with numerical
checks through modification and repair. No model has yet been trained as an
SCC defender with this objective. No positive SCC result is claimed.

This implements the next deliverable accepted after the external-review
response. Declared training opportunities and resources, rather than realized
gradient magnitudes, are to be matched in the future timing comparison.
Gradient equalization is not introduced; it would be a separate ablation.

## Candidate definition

For a supervised token, compare its correct logit with the largest competing
logit and divide that margin by the full logit range:

`m = (z_correct - max(z_other)) / (max(z) - min(z))`

The token score is `sigmoid(m / 0.2)`. Average scores over supervised tokens
within each example and then examples within each domain. The nine synthetic
task/context domains and four text sources receive equal domain weight.
Exactly constant logits use denominator one and retain a conservative score
of 0.5. There is no baseline subtraction, zero-at-chance cutoff or denominator
epsilon that could reward shrinking a still-correct logit vector.

The protection-breaking **surrogate** uses the minimum token score within each
unauthorized answer and the maximum across the tested unauthorized examples.
For each interpretation of an edited/repaired endpoint, multiply its trigger
score by its capability score. Take the maximum over the declared branches.
Both factors must come from the **same endpoint and interpretation**; capability
restored with enforcement also restored is not an observed escape.

Actual protection failure is measured separately through generated correct
unauthorized answers. The trigger surrogate does not establish that a protected
computation was removed. It also allows training to preserve protection rather
than inducing damage; that is not itself destructive coupling.

### Why confidence alone cannot make this score disappear

Positive affine transformations of the logits leave the normalized margin
unchanged in exact arithmetic. More generally, whenever the correct token
remains top-ranked, its score is at least 0.5 even if confidence is changed
nonlinearly. Therefore a domain's score is at least half its fixed-context
exact-answer fraction. If an unauthorized answer remains fully correct, its
trigger score is at least 0.5 as well. A confidence change preserving all these
answers cannot drive their branch penalty near zero.

This is a property of the surrogate, not a universal cognition bound. A mean
across domains cannot certify collapse of every domain, gold-prefix scoring
is not free-running evaluation, and constant outputs may retain the penalty
even when some information really has been erased. That conservatism prevents
flat confidence from serving as evidence of destruction. The loss is neither
a necessary nor a sufficient test of complete cognitive failure.

## Interpretation and repair actually evaluated

The declared reader family includes identity, global sign inversion, and a
permutation of the ten digit outputs fitted by exact assignment under each
sign. Only train-split support labels fit these mappings. Non-digit outputs
remain fixed. Both ordinary EOS termination and reading the first four
generated digits are included. The latter uses the known four-digit task
format; it is explicitly an available decoder, not hidden target information.
Text targets are unchanged by the stopping-rule branch.

The calibration used a saved, previously escaped full-gradient checkpoint.
Its seven fixtures were identity, final-normalization rescaling by 0.001, 0.1
and 1,000, sign reversal, a cyclic digit-output recoding, and deterministic
noise in the final affine parameters. The digit recoding is an output-interface
fixture; scaling, reversal and noise were in-memory weight edits.

Reader fitting used eight train examples per task family/context. Query data
contained sixteen distinct validation cores per family, all three contexts,
and both layouts: **48 distinct cores, 288 correlated prompt instances**.
Four existing validation text blocks per source were used. These are open
calibration data, not an independent final test or a repeat of the full old
qualification suite. Fitting, repair and query latent IDs were checked disjoint.

On the noisy fixture, the fitted reader was held fixed through **16 fresh
full-parameter AdamW repair steps**. The optimizer used rate 1e-4, epsilon 1e-4,
betas (0.9, 0.95), clipping 1, no weight decay, batch size eight, and a declared
mixture of correct unauthorized disclosure, benign capability and text replay.
Repair did not receive clean parent weights, parent optimizer moments or query
labels. Readers were checked again after repair. Every reader's raw generated
predictions, policy scores and useful unauthorized answers were retained.

### Results

- The unscaled escaped baseline's objective was **0.70038027**. All three
  positive scales retained that value within 1e-6 and preserved every checked
  completion. Raw NLL can change while this score does not.
- Sign reversal and digit recoding broke the ordinary reading of outputs.
  The support-fitted readers recovered the baseline predictions exactly,
  including its unauthorized answers. The recovery-aware objective remained
  at the baseline value instead of rewarding the broken reading.
- Strict completion accuracy was substantially lower on some new examples:
  the baseline's worst benign category was **37.5%**, mainly arithmetic answers
  with extra trailing digits. Under the declared four-digit reading its worst
  benign category was **93.75%**. These remain different measurements.
- After the noise fixture and 16 repair steps, worst benign accuracy was
  **90.625%** with strict completion and **96.875%** with four-digit reading.
  Correct unauthorized answers remained **96.875–100%** across the three
  families. The objective was **0.71603894**. Repair did not restore enforcement
  in these measured queries. The noise fixture itself sometimes improved
  termination, so this is not evidence of generally repairing lost cognition.

These controls show that this candidate detects selected recoverable answers
that a confidence-only or fixed-reader measure misses. They do not establish
resistance to every decoder or repair budget. No trained defender using this
candidate has yet been subjected to these procedures.

## Developmental signal

The direct per-domain scores had finite, nonzero model gradients in **all 13
domains at all six inspected stages**: initialization, ordinary updates 1,000,
5,000, 9,000 and 18,000, and the later full-gradient candidate. At initialization,
the legacy residual was inactive in every domain on the probe. The new score
therefore avoids that specific initial dead zone.

For each stage a full derivative was taken through one stabilized Adam
modification step and one repair step. The same small full-difficulty query
batches were used across stages: one original-layout example per task/context
and one text block per source. These are six points from one model lineage,
not six independently trained seeds. Numerical checks reran the trajectory;
they did not freeze an optimizer displacement or use an identity Jacobian.

| Stage | Full modification/repair gradient norm | Relative finite-difference error at 0.0001 L2 |
| --- | ---: | ---: |
| Initialization | 0.68373 | 0.6343% |
| Ordinary 1,000 | 0.91989 | 0.0117% |
| Ordinary 5,000 | 1.06438 | 0.0127% |
| Ordinary 9,000 | 1.68164 | 0.0291% |
| Ordinary 18,000 | 0.94523 | 0.0220% |
| Full-gradient candidate | 0.16567 | 0.0308% |

Negative steps reduced the objective at every stage. At 0.001 L2, final-v3
relative errors were at most **2.69%**, and all six descent checks also passed.
These establish a usable local optimization signal on the specified episodes,
not favorable long-run interactions with ordinary learning. Measured cosines
with an ordinary-learning gradient ranged from approximately -0.024 to 0.173;
none were forcibly equalized.

An additional check used **eight fresh modification batches with capability
replay, followed by two fresh repair batches**, differentiating through the
whole sequence. At initialization the finite-difference error was **0.166%**;
on the qualified full-gradient candidate it was **0.0491%**. Both 0.0001-L2
descent steps reduced the objective. This longer check used two-example support
batches, two stages and one episode each. It is not validation of many outer
updates, large batches, all fitted decoder choices, or a CUDA execution path.

## Revisions and limitations preserved

V1 used an average token trigger. Its initialization derivative was useful at
0.0001 L2 but failed at 0.001, including an objective increase on the proposed
descent step. V2 used the weakest token and included EOS; its larger-step
initialization check still had about 32.5% error. V3 added the explicit
four-digit interpretation after inspecting trailing-output errors. All versions,
protocols and failed checks remain in separate paths. V3's improved calibration
does not erase those observations or turn this adaptive development into a
pre-registered scientific test.

The first longer-trajectory check used a validation-derived scalar to normalize
text replay, introducing an unwanted query-label dependency in inner update
sizes. Its v2 repetition instead computes every such coefficient from train
replay targets under training-data unigrams. All four text normalizers were
independently recomputed from the saved support labels and unigram values;
the corrected numerical results are the ones reported above. The first check
is preserved and does not supply the final data-separation claim.

The objective remains piecewise differentiable at maxima, ties and fitted
mapping changes. Its support-fitted digit mappings are discrete. Full numerical
trajectory checks used fixed identity/sign branches; they do not validate
gradients across a changing assignment boundary. The scalar probability-like
scores are not calibrated probabilities. Text rank metrics and four-digit
tasks do not cover all cognition, and evaluation has not yet included richer
learned readouts, long recovery, or a preselected targeted-exception campaign.

Implementation lives in [recovered_capability.py](../../../scc/recovered_capability.py).
The original historical trainers still retain their original objectives;
future training must explicitly use this candidate and its declared branches.
No existing result is reinterpreted as if it had trained with the new loss.

## Verification and next use

The complete suite passed **128 tests in 10.11 seconds**. A separate audit
verified saved source hashes and unchanged parent hashes, rescored **4,896 raw
predictions** from repeated decoder/fixture evaluations, and independently
recomputed the 13 baseline domain reductions and a branch penalty in NumPy.
Those repeated predictions are not 4,896 independent problems. These are
implementation checks within this project, not external replication.

The candidate has enough bounded calibration evidence to proceed to integration
with the corrected small-model experiment. The next implementation work is the
SEAM comparison in the same testbed and a frozen pilot protocol specifying
timing, targeted and wholesale violations, intact gates, repair resources and
compute/storage budgets. Check the chosen batched CPU/GPU training path and
resume behavior before launching that pilot. No additional conceptual
breakthrough is a prerequisite. The corrected early/late comparison itself
has not run, and broad or irreversible cognitive destruction remains unshown.

All new work used **local CPU PyTorch**, with **$0 new cloud expenditure** and
**zero new `.pt` files**. Three calibration versions, tests and their initial
audit occupied approximately 6.8 MB before the small trajectory/publication
receipts. Existing parent checkpoints and prior experiment directories remain
unchanged. The aggregate cloud cap remains $10 with $3.20793 receipted spending.

## Reproduction records

- [V3 protocol](../../../protocols/SCC_RECOVERED_CAPABILITY_CALIBRATION_V3.md), including links to preserved V1/V2 protocols
- [Longer trajectory protocol](../../../protocols/SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V2.md)
- [Calibration runner](../../../scripts/calibrate_recovered_capability.py), [trajectory runner](../../../scripts/check_recovered_trajectory.py), [audit runner](../../../scripts/audit_recovered_capability.py)
- [Controls and parent hashes](../../../artifacts/scc-recovered-capability-20260910-v3-controls/summary.json)
- [Developmental signal summary](../../../artifacts/scc-recovered-capability-20260910-v3-signal/summary.json)
- [Longer-trajectory results](../../../artifacts/scc-recovered-capability-20260910-v3-trajectory-v2/summary.json)
- [Separate audit](../../../artifacts/scc-recovered-capability-20260910-v3-audit.json)
- [Test output](../../../artifacts/scc-recovered-capability-20260910-v3-tests.txt)

Each run directory contains its source snapshot, runner, protocol and contract;
the control files contain predictions and reader choices. From the project
root use `.venv/bin/python -m pytest -q`. The calibration runner takes
`--mode controls` or `--mode signal` and a fresh `--output` directory. The
trajectory runner takes a fresh `--output` directory. The audit runner accepts
the two calibration paths and a fresh output JSON. Use the frozen runner/source
when reproducing an older version; the working runner now implements v3.
