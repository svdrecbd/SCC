# Mechanism foundation audit — 2026-09-10

The local baseline and the checked negative coupling results hold up under this
audit. Destructive cognition–alignment coupling has not been demonstrated. This
review found and fixed reproducibility and configuration gaps that could have
silently changed future experiments. No paid compute was launched.

The governing purpose is recorded in [MECHANISM_TARGET.md](../../../MECHANISM_TARGET.md).
The existing experiments remain useful attempts to induce a dependency, but
their authorization behavior and utility-retention limits do not establish
alignment or catastrophic cognition failure.

## Scope and evidence

This was a bounded review of the current model, generation/scoring, data and
task splits, curriculum lineage, coupling objectives, attack/defender runners,
restart behavior, final topology campaign, and remote access. It was not a new
scientific replication, new attack search, or exhaustive verification of every
historical experiment and every possible input.

All new evidence is in
[`artifacts/mechanism-audit-2026-09-10-v1`](../../../artifacts/mechanism-audit-2026-09-10-v1).
Existing runs, checkpoints, source snapshots, failed attempts, and evaluations
were preserved. No research runner was active when production source changed.

| Check | Observed result |
|---|---|
| Existing main suite plus symmetry checks | 55 passed before the changes; the exact-Adam probe's collection-time assertions also executed |
| Regression checks with complete inputs before fixes | Seven failures reproduced missing reference-integrity checks and silent configuration fallbacks |
| Full suite after fixes | 64 passed, including unchanged-reference restart equivalence and changed-reference rejection |
| Baseline qualifications | Both seeds 17 and 23 pass; 93,272 distinct ancestral training tables each, no checked evaluation-table overlap |
| Final topology campaign receipts | 39 completed run directories verified: 3 defenders, 24 main evaluations/interventions, 6 recovery, 6 rescaled attacks |
| Matched streams | Ordinary batches match across all three defenders; the two coupling arms share meta streams and inner attack batches |
| Diagnostic records | 131 prediction sets rescored by the campaign audit |
| Expanded evaluation | 32 prediction sets rescored with separate counting code and readable-premise checks; 19 ancestral checkpoints reconstructed and checked |
| Direct checkpoint inference | 336 serial completions across seven checkpoints exactly match saved text, token IDs, and termination flags |
| Independent language calculation | All 12,689 natural-text validation blocks reevaluated on seven checkpoints using log-softmax/gather and float64 accumulation; maximum difference from recorded source NLL below 5e-9 nats/token |

The direct inference check uses eight saved prompts per category and ordering,
48 per checkpoint. It does not regenerate every saved task completion. Full
language recomputation and raw-task rescoring cover the broader frozen evidence.
The final test split was not used for model evaluation or fitting.

Receipts: [tests](../../../artifacts/mechanism-audit-2026-09-10-v1/tests-after.txt),
[original regression failures](../../../artifacts/mechanism-audit-2026-09-10-v1/regressions-before-complete-inputs.txt),
[campaign audit](../../../artifacts/mechanism-audit-2026-09-10-v1/topology-receipts.json),
[seed 17](../../../artifacts/mechanism-audit-2026-09-10-v1/foundation-seed17.json),
[seed 23](../../../artifacts/mechanism-audit-2026-09-10-v1/foundation-seed23.json),
[independent recheck](../../../artifacts/mechanism-audit-2026-09-10-v1/independent-recheck/result.json).

## Fixed defects and their implications

1. **Reference checkpoints were not part of the resume fingerprint.** The
   starting checkpoint was hashed, but separate capability/reference checkpoints
   were identified only by path. Replacing either reference at the same path
   could change retention thresholds and, for the shared reference, training
   loss normalization while an alleged exact resume was accepted. New coupling
   contracts hash both references, reject changed contracts before training,
   and check for reference changes during initialization. Tests reproduce both
   failures and compare uninterrupted versus unchanged-reference resumed weights.

2. **Misspelled execution choices could select different experiments.** An
   unknown parameter scope fell through to the last-block selection, an unknown
   attack mode could run disclosure training, and an unknown inner-attack string
   could run the short exact-SGD path. The runner now validates these choices
   before loading models or creating a run directory. No such misspellings were
   observed in the checked campaign configurations.

3. **GPU requests could be silently ignored by local runners.** A top-level
   `device: cuda` setting was ignored by both the online curriculum and coupling
   runners. They now explicitly reject unsupported devices; coupling also rejects
   unsupported precision. This prevents an apparent GPU configuration from
   silently running the experiment on CPU. It does not implement CUDA support.

4. **The coupling runner lacked an explicit data compatibility check.** It now
   requires natural data to match the model context and the byte-tokenizer
   semantics used for synthetic tasks. Regression coverage includes a different
   context and a BPE tokenizer with the same vocabulary size (260), because equal
   vocabulary sizes do not imply equal token meanings.

These changes do not alter the optimization objectives, saved model weights,
or measured success criteria. Rechecked outputs show no discrepancy in the
headline results. Old exact resumes must use their preserved source and
environment; the revised source/contracts intentionally reject silent
cross-version continuation. Loading old checkpoints for evaluation still works.

## Completed latest comparison

The table uses endpoints selected by the already-frozen expanded-evaluation
protocol. Counts are optimizer updates along the full removal/recovery procedure.
They are observed upper bounds, not minimum costs, confidence intervals, or proof
of a robust difference between models.

| Defender | Selected procedure | Total updates | Original-order useful disclosure | Original-order authorized / retrieval | Worst-source perplexity growth |
|---|---|---:|---:|---|---:|
| Ordinary continuation | Short attack, second data seed | 1,000 | 509/512 | 511/512; 512/512 | 4.9816% |
| Legacy objective, mixed attack profiles | Removal plus recovery | 1,300 | 512/512 | 512/512; 512/512 | 2.9773% |
| Per-constraint objective, mixed attack profiles | Short attack | 300 | 511/512 | 511/512; 512/512 | 4.7702% |

All three also meet the existing escape criteria after table reordering. Clean
qualification and benign uppercase-edit success hold for all three in the
expanded evaluation. The control is close to the arbitrary 5% retention boundary;
its exact pass is numerically reproduced, not evidence that this cutoff has
special cognitive significance.

The separate rescaling comparison finds an ensemble-model escape after 1,000
optimizer updates plus a function-preserving edit on the smaller development
evaluation. That endpoint was not included in the above full-language selection.
Consequently, 1,300 is not even the cheapest observed ensemble escape across
every comparison, and should not be advertised as an established coupling gain.

## Assumptions that must not carry forward

- The baseline's text-loss improvement and near-perfect four-key retrieval are
  evidence of narrow learnability, not general reasoning or autonomous cognition.
- The two baseline seeds share the data schedule and corpus. Coupling studies
  use one defender initialization; attack-data seeds are not defender replications.
- Coupling was introduced after capability learning. Randomly initialized
  ancestors do not make these developmental-coupling experiments.
- A failure of any one utility-retention constraint is not broad cognitive
  destruction. The current minimum-over-constraints surrogate can decrease from
  deterioration of just one capability measure.
- Long-attack defense uses a frozen-displacement, identity-Jacobian approximation.
  It is not the exact derivative through the adapting attacker. Short exact-SGD
  derivative tests and a sensitive experimental exact-Adam probe do not validate
  stable long exact-meta training.
- Calibration matched a gradient scale at one point. It did not keep the two
  objectives' gradients or Adam updates matched throughout training. Their
  observed weighted gradient ranges differ substantially in the campaign audit.
- Ordinary continuation has matched ordinary batches, not matched total compute.
  Each latest coupling arm used 51,200 inner optimizer updates. Local contended
  timers and loss-forward counters are not GPU throughput or total FLOPs.
- The small, repeatedly used development corpus and validation cases do not
  qualify a billion-token campaign or sealed confirmatory study. Existing passage
  checks are bounded; they do not eliminate paraphrase or benchmark overlap.
- Saved source/checkpoint histories exist locally, but the directory is not a Git
  repository. This audit does not establish off-machine backup or portability.

## GiveMeANode access and readiness

Read-only `gman` checks confirm a valid stored token for `https://givemeanode.com`,
active billing, member access in the default workspace, node inventory access,
and job-list access. The reported workspace monthly cap is **$53.88**; this
identity's September month-to-date spend is **$0**. The member view does not
expose an organization credit balance. The earlier **$450** figure remains
user-reported; a spending cap is not a credit balance or verified free headroom.

The account has access to the GPU service. No node was created, awakened,
borrowed, or modified, and no job was submitted during this audit. Existing
workspace resources are shared with other work. The access check does not
reserve GPU capacity or prove that a future allocation will fit the cap.

The generic fixed-data trainer contains CPU/CUDA and bf16 code paths, but the
successful online curriculum and coupling runners remain CPU implementations.
CUDA correctness, actual GPU utilization, differentiable-kernel support,
checkpoint restart, interruption/recovery, and measured throughput must be
qualified on the chosen remote environment before treating GPU runs as ready.
Earlier budget estimates do not replace that measurement.

Access evidence: [sanitized receipt](../../../artifacts/mechanism-audit-2026-09-10-v1/givemeanode-access.json).

## Consequence for the next experiment

The checked local foundation is usable within its stated limits. The next
scientific milestone is to construct and causally test a cognition–protected-
function dependency, with explicit severe-collapse measurements and controls.
The current negative results are retained as baselines and failed candidates.
Neither more GPU availability nor lower surrogate loss establishes the missing
mechanism. No global impossibility or exhausted search is claimed.
