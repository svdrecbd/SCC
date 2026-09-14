> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Audited corpus workflow

The current implementation selects a bounded development sample of existing
Common Pile text, combines it with our authorization and capability tasks, trains
our own tokenizer, and runs a baseline qualification check. The approximately
50M, 1B-token pilot and its final data expansion remain separate gates.

## Run the working local foundation

The [recovery report](../../../reports/RETRIEVAL_RECOVERY.md) is the current result.
The successful path uses a byte vocabulary and our own curriculum checkpoints;
the earlier BPE qualification commands below reproduce the historical failure.

Prepare the frozen corpus in bytes once, if this directory does not already exist:

```sh
uv sync --locked --extra dev
uv run --locked python -m scc.data artifacts/corpus-qualification/recipe/records.jsonl artifacts/retrieval-recovery/byte-prepared --context-length 192
```

Then run the complete curriculum from random initialization in a fresh directory:

```sh
uv run --locked python scripts/run_foundation.py --output runs/my-foundation --seed 17
```

This runs lookup, four-character copying, original-format authorization, and
mixed natural-text training. Each stage must pass its behavioral gate before
the next starts. The final check also requires improvement over the per-source
unigram and pre-mix model on held-out natural text. Failure preserves the run
for inspection. It uses local CPU compute and creates no cloud job.

For additional development tables after completion:

```sh
uv run --locked python -m scc.online_evaluate --checkpoint runs/my-foundation/04-mixed/step-00008000.pt --output runs/my-foundation/additional-validation.json
uv run --locked python scripts/verify_foundation.py --run runs/my-foundation/04-mixed --additional runs/my-foundation/additional-validation.json --output runs/my-foundation/verified-with-additional.json
uv run --locked --extra dev python -m pytest -q
```

Configurations, checkpoint parents and hashes, exact source snapshots, RNG and
optimizer state, source exposure, raw predictions, and table identities are
preserved. Curriculum stages intentionally revisit some earlier training tables;
whole-table hash partitions keep the complete lineage separate from validation.
The final mixed stage uses a new data seed. Final test data remain unused.

## Reproduce the original BPE qualification in a fresh output directory

Use the isolated `uv` environment (Python 3.14.7, PyTorch 2.14.0). The first two
mixed-baseline runs used Python 3.12.7 / PyTorch 2.12.0; the original environment
is recorded in `artifacts/corpus-qualification/environment.json`, and each run
retains its source snapshot. A newer tokenizer/library version need not reproduce
an older artifact byte for byte. Use the frozen tokenizer and prepared data for
comparisons, and saved source/environment when reproducing an old build.
No command below creates a cloud job.

```sh
uv sync --locked --extra dev
uv run --locked python -m scc.corpus acquire --config configs/corpus_qualification.json --output artifacts/reproduction/acquired
uv run --locked python -m scc.corpus audit --acquired artifacts/reproduction/acquired --output artifacts/reproduction/audited
uv run --locked python -m scc.recipe --audited artifacts/reproduction/audited --output artifacts/reproduction/recipe
uv run --locked python -m scc.leakage --records artifacts/reproduction/recipe/records.jsonl --output artifacts/reproduction/leakage.json
uv run --locked python -m scc.tokenizer --records artifacts/reproduction/recipe/records.jsonl --output artifacts/reproduction/tokenizer --vocab-size 4096
uv run --locked python -m scc.data artifacts/reproduction/recipe/records.jsonl artifacts/reproduction/prepared --context-length 192 --tokenizer artifacts/reproduction/tokenizer
uv run --locked python -m scc.qualify --config configs/qualification_long.json --data artifacts/reproduction/prepared --records artifacts/reproduction/recipe/records.jsonl --output runs/reproduction --threads 4
```

Acquisition resumes completed, checksummed shard prefixes. Auditing, recipe
creation, tokenizers, preparation, and training use fresh output paths to preserve
previous evidence. Source snapshots inside each run preserve the version actually
executed. Final model checkpoints include optimizer and sampler/RNG state.

The current passage matcher checks all earlier documents sharing an anchor.
Compared with the original frozen audit used in the reported runs, it retains the
same text and changes four document split assignments after merging related
groups. The new audit is saved as `artifacts/corpus-qualification/audited-all-pairs`.
Use the source snapshot in `artifacts/corpus-qualification/recipe/source` to
rebuild the original audit exactly. The commands above exercise the current
implementation and therefore do not recreate the historical split bytes.

## Source selection and exclusions

The source revisions are pinned in `configs/corpus_qualification.json`.
Acquisition scans 30,300 records across 11 selected shard prefixes. It does not
download entire multi-terabyte datasets. The reviewed sample is restricted to:

- Wikipedia content pages from Common Pile's Wikimedia collection;
- openly licensed PressBooks chapters, grouped by book;
- openly licensed LibreTexts chapters, grouped by book;
- Gutenberg works up to 200,000 characters, kept whole and grouped by title.

Record-level metadata retain the declared original licenses and provenance.
The normalized allowlist includes public-domain declarations and CC BY / CC BY-SA
URLs; it excludes other or unrecognized declarations. This is not independent
legal verification. Upstream dataset cards are saved with the acquired records.

Filtering removes noncontent namespaces, short or unusually long documents,
some extraction/encoding damage, low text diversity, and benchmark-name matches.
Normalized exact duplicates are removed across the scanned sources. Work IDs
and detected shared 32-word passages form connected groups before splits are
assigned. An independent checker verifies selected-data identities, exact content,
and the same passage criterion across every split. This still misses paraphrases,
short answer overlap, and unknown benchmarks.

## Mixture, masks, and metrics

The language documents use next-token prediction. The generated tasks supervise
the completion and EOS, with prompt context available to the model. The BPE
tokenizer is trained solely on training records and has no imported learned
vocabulary. Control-token IDs are separate from the learned byte-level vocabulary.

Each training batch is sampled from a configured source/category. These are batch
probabilities, not token fractions. The trainer logs actual input-token,
supervised-token, and example exposure for every source. Trailing unused padding
is trimmed for computation. These exposure counters, sampler order, and RNGs
survive an exact CPU resume.

Qualification evaluates a fixed development validation slice before and after
training, with a separate training-fitted unigram for each source. Synthetic
behavior is scored on 64 deterministic IID validation examples per category.
EOS termination and the correct answer are required. Incorrect or empty answers
do not count as useful disclosures. Model evaluation does not use the test split.

The 2,500- and 20,000-update mixed runs both failed the original four-character
retrieval task. A deeper, task-only run and two simpler calibration runs are
listed in `configs/qualification_experiments.json`. All start from our random
initialization. These are exploratory checks, not claim-bearing SCC experiments.

The one-character calibration changes the value length while retaining the
permission rule and four-key lookup. It uses the frozen original tokenizer and
ordinary-text sample. It must not replace the harder task in reported results:

```sh
uv run --locked python -m scc.recipe --audited artifacts/corpus-qualification/audited-v2 --output artifacts/calibration-reproduction/recipe --value-length 1
uv run --locked python -m scc.leakage --records artifacts/calibration-reproduction/recipe/records.jsonl --output artifacts/calibration-reproduction/leakage.json
uv run --locked python -m scc.data artifacts/calibration-reproduction/recipe/records.jsonl artifacts/calibration-reproduction/prepared --context-length 192 --tokenizer artifacts/corpus-qualification/tokenizer
uv run --locked python -m scc.qualify --config configs/qualification_calibration_deeper.json --data artifacts/calibration-reproduction/prepared --records artifacts/calibration-reproduction/recipe/records.jsonl --output runs/calibration-reproduction --threads 4
```

## Rebuild the saved report

The report reads the original audit, rebuild checks, and every registered run:

```sh
uv run --locked python scripts/report_qualification.py --corpus artifacts/corpus-qualification --run runs/corpus-qualification-v2 --output reports/CORPUS_QUALIFICATION.md
```

## Coupling and intervention development

The [coupling report](../../../reports/COUPLING_DISCOVERY.md) records a validated
implementation and a failed first candidate. These local commands use the saved
seed-17 foundation checkpoint and byte-prepared corpus. Choose fresh output
paths; historical results are never overwritten.

```sh
uv run --locked --extra dev python -m pytest -q
uv run --locked python scripts/check_coupling_gradients.py --output artifacts/coupling-reproduction/gradients.json --epsilons 0.000001 0.0000001 0.00000001
uv run --locked python -m scc.coupling_run --config configs/coupling/defense17_escape.json --output runs/coupling-reproduction-escape
uv run --locked python scripts/run_coupling_matrix.py --config configs/coupling/matrix17.json --output runs/coupling-reproduction-matrix
uv run --locked python scripts/report_coupling.py --output artifacts/coupling-reproduction/audit.json
```

The matrix configuration points at the three **saved** defenders. To attack
newly reproduced defenders, first reproduce all three defense configurations,
then copy the matrix configuration and update its `arms` checkpoint paths.
The report script audits the original discovery directories; it does not
automatically substitute reproduction runs. An exact historical rerun must use
the source and environment snapshot saved in that run. Later source changes
intentionally change the run contract even when behavior is unaffected.

`--stop-after` and `--resume` on `scc.coupling_run` support exact restart into a
fresh directory with an unchanged configuration and source contract. Benign
policy-preserving follow-ups use `configs/coupling/benign_policy17_*.json`;
they are separate from the initial matrix's failed benign controls.

## Stronger-attacker follow-up

The [latest report](../../../reports/STRONG_ATTACK_RESULTS.md) records the stronger
first-order training attack, two coupling strengths, shorter external bypasses,
full validation coverage, and the separate exact-gradient investigation. Read
the [protocol](../../../reports/STRONG_ATTACK_PROTOCOL.md) and
[working standard](../../../WORKING_STANDARDS.md) before extending the campaign.

These commands use saved seed-17 checkpoints and fresh output paths:

```sh
uv run --locked python scripts/calibrate_strong_attack.py --config configs/strong-attack/calibration17.json --output artifacts/strong-reproduction/calibration
uv run --locked python scripts/check_strong_gradient.py --output artifacts/strong-reproduction/gradient-check.json
uv run --locked python -m scc.coupling_run --config configs/strong-attack/defense17_escape.json --output runs/strong-reproduction-escape
uv run --locked python scripts/run_short_comparison.py --arms control refusal escape escape_weight10 --output-prefix runs/strong-reproduction-short
uv run --locked python scripts/evaluate_strong_endpoints.py --config configs/strong-attack/matrix17.json --output artifacts/strong-reproduction/evaluation --blocks-per-source 0
uv run --locked python scripts/report_strong_attack.py --config configs/strong-attack/matrix17.json --output artifacts/strong-reproduction/audit.json
```

The short-comparison helper targets the four **saved** stronger defenders.
The broader-evaluation helper reads defender and long-attack roots from the
configuration, but discovers short endpoints at `runs/strong-short-defender17-*`.
The audit scans the saved campaign paths. Consequently these last three commands
verify/re-attack historical models; they do not automatically evaluate a newly
reproduced defender. For a new campaign, copy the configurations and helpers and
explicitly update every defender, attack, and reference path before execution.
Use the run's saved source/environment snapshot for exact historical reproduction.

`inner_attack: full_adamw_first_order` uses native copied-model AdamW for its
forward trajectory and holds the resulting displacement fixed for the outer
derivative. It does not differentiate through 1,300 optimizer steps. Experimental
long derivatives under `experiments/` have separate source copies and numerical
receipts; they are not the production defense-training path. The results report
records their small-scale agreement and severe long-trajectory conditioning.

The current implementation is intentionally bounded. Scaling intake to hundreds
of millions of tokens requires disk-backed candidate selection/deduplication and
a broader source/shard sample; it should not load the entire corpus into RAM or
repeat the small qualification pool until a nominal billion-token counter is met.
