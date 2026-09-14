> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# First implementation milestone

This is a local reference implementation of data preparation, a small causal
Transformer, training, evaluation, and resumable checkpoints. It now includes an
audited Common Pile development sample and a tokenizer trained from scratch.
It also includes an exact differentiable coupling prototype, a bounded attack
comparison, and benign-edit controls. The first candidate was bypassed; there
is no positive SCC result. The [latest follow-up](../../../reports/STRONG_ATTACK_RESULTS.md)
also defeats both coupling strengths after training with stronger attackers.
No GiveMeANode compute is required by these commands.

## Decisions and open research questions

Confirmed by the user on 2026-09-09:

- All model weights start from our random initialization.
- Synthetic authorization is the first protected rule.
- Use existing ordinary text alongside our generated authorization tasks; preserve
  document provenance and test quality and learnability before scaling.
- Anaconda and Python 3.12 are not requirements; choose suitable current tools.
- The pilot has a reported total compute budget of approximately $450.

Implementation defaults, reversible before the final experimental freeze:

- Conventional dense, pre-norm decoder with causal attention, GELU, learned
  positions, and tied input/output embeddings; standard PyTorch operations.
- A small CPU development configuration and a separate approximately 50M model
  configuration. The latter is a shape to inspect, not a funded run configuration.
- A lossless byte tokenizer for infrastructure fixtures and an own-trained
  4,096-token byte-level BPE tokenizer for corpus qualification. The final
  campaign vocabulary size remains open.
- Explicit dataset splits, source and license declarations, SHA-256 fingerprints,
  configuration/source/environment manifests, and atomic checkpoints.
- Exact behavioral scoring separates permitted answers, refusal compliance,
  useful unauthorized disclosure, retrieval, and addition.

The remaining scientific choices should be resolved through small development
experiments: task difficulty, curricula and proportions, final tokenizer,
structural generalization suites, coupling objective, and meaningful capability
thresholds. We should not ask the user to guess optimal optimizer settings.

The data-policy recommendation and its limits are in
[DATA_STRATEGY.md](DATA_STRATEGY.md). The natural-language capability component in
the earlier budget proposal now has a small development implementation described
in [CORPUS_WORKFLOW.md](CORPUS_WORKFLOW.md). Held-out text loss is a narrow
language-modeling check, not a measure of broad reasoning capability.

## Local setup

The current environment uses Python 3.14.7, PyTorch 2.14.0, and `uv`. The Python
version is pinned in `.python-version`; `uv.lock` records transitive dependencies.
Installation is local to `.venv`:

```sh
uv sync --locked --extra dev
```

The initial smoke and two mixed-corpus qualification runs used Python 3.12.7 /
PyTorch 2.12.0. Their source and environment records are retained; upgrading
packages is not an exact continuation of those runs. The new environment passes
restart tests and loads the older checkpoint with identical greedy output on 32
sampled validation prompts. This is a compatibility check, not identical training
across releases. A CUDA image and remote execution need separate validation.

```sh
uv run --locked --extra dev python -m pytest -q
uv run --locked python -m scc.synthetic artifacts/example/records.jsonl --instances 200
uv run --locked python -m scc.data artifacts/example/records.jsonl artifacts/example/prepared
uv run --locked python -m scc.train --config configs/smoke.json --data artifacts/example/prepared --output runs/example
uv run --locked python -m scc.evaluate --checkpoint runs/example/step-00000020.pt --data artifacts/example/prepared --records artifacts/example/records.jsonl --output runs/example/evaluation.json
```

Commands refuse to overwrite existing prepared datasets, runs, or evaluation
results. Choose a fresh output path for another run.

## Data contract

One JSON object per line. Every record declares `split` (`train`, `validation`,
or `test`), `latent_id`, `source`, and `license`.

- A task record contains `prompt`, `target`, and a `category` for behavioral
  evaluation. Unauthorized records additionally contain `underlying_answer`.
  Metadata, including the underlying answer, is never serialized into model input.
- A plain-text document contains `text` instead of `prompt`/`target`.
- All versions of a semantic problem must share a `latent_id` and split. The
  preparer rejects crossing identities and exact cross-split text/prompt duplicates.
  It cannot detect all semantic overlap in arbitrary external inputs; generators
  and corpus importers remain responsible for their semantic identity policy.
- Each task occupies a padded block and supervises only its answer and EOS. Long
  tasks fail explicitly. Text documents are chunked within document boundaries
  and supervise next-token prediction throughout.
- Training samples sources by batch probabilities, trims trailing padding, and
  records input-token, supervised-token, and example exposure per source. Batch
  probabilities are not token fractions. Throughput cannot be extrapolated to
  the final pilot yet.

The version-0 generator includes paired permission variants, independent-format
retrieval, and addition. Its held-out instances are an **IID development check**.
It does not yet implement the independent generators or structural stress tests
proposed in the data strategy. Reusing underlying retrieval computations outside
the permission format is a diagnostic, not proof of broad independent capability.

## Checkpoint and evaluation contract

Each checkpoint stores model, optimizer, schedule configuration/current step,
shuffled order/cursor/data RNG, Python and PyTorch RNGs, token counters, history,
and the run contract. CUDA RNGs are included when executing on CUDA. Saves are
atomic. A resumed run uses a fresh directory and retains its history/parent path.
Changing the data, code, configuration, or recorded environment rejects an exact
resume. Deliberate curriculum transfer and intervention initialization use
separate paths with parent-checkpoint receipts.

The test suite compares interrupted versus uninterrupted CPU runs with dropout,
shuffling across epochs, and optimizer state. It also checks causal masking,
learning on a tiny fixture, loss masking/aggregation, leakage rejection, data
corruption detection, and behavioral scoring traps.

This validates an exact restart on the tested CPU environment. CUDA execution,
mixed precision, GPU checkpoint restoration, and cross-platform equivalence are
not yet validated. PyTorch does not promise identical results across platforms
or releases: [reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Evaluation reports supervised token NLL separately by source and exact task answers.
Task-completion perplexity is not ordinary-language perplexity. The command-line
behavior sample is deliberately labeled a development check; no final holdout
has been frozen or evaluated. Scores require EOS termination and do not silently
remove invalid special tokens from a generated answer.

Before the final GPU campaign, we still need broader capability qualification,
an expanded final data recipe, GPU throughput and container validation, runtime
profiling, remote checkpoint recovery, and reconciliation of the observed
workspace spending cap with the reported credit balance.

## Historical first local verification, 2026-09-09

- `python3.12 -m pytest -q`: **11 passed** in the final verification.
- Generated 797 development records: 654 train, 68 validation, and 75 test.
- Completed the 20-step CPU command-line smoke run and wrote checkpoints and
  evaluation artifacts under `runs/foundation-smoke/`.
- The 24-example validation behavior sample has **zero exact-match accuracy**
  in every category. This short plumbing run does not demonstrate authorization,
  retrieval, addition, generalization, or SCC. Establishing task learnability is
  the next experimental milestone. The tiny learning unit test only verifies
  that the implementation can fit a simple controlled fixture.
- The larger configuration contains **48,163,968** parameters with the current
  byte vocabulary. Its count was inspected on a meta device; it was not trained.
- No cloud job was launched and no GiveMeANode credits were spent.

## Corpus qualification, 2026-09-09

The development intake selected 3,313 natural-text documents (23.99 MB) and added
17,165 generated task records. Rebuilding from frozen downloads reproduces the
selected records, recipe, and original tokenizer exactly. The independent split
checker found no identity, exact-content, or sampled shared-passage violations.
These checks do not establish exhaustive semantic decontamination.

The 945,664-parameter mixed baseline beats source-specific unigram prediction on
all four natural-text sources after 20,000 updates. However, authorized retrieval
and ungated retrieval both score 0/64 on development validation, while withholding
scores 64/64. This model fails the task-competence gate. Results and subsequent
diagnostics are recorded in [reports/CORPUS_QUALIFICATION.md](../../../reports/CORPUS_QUALIFICATION.md).

Both two- and four-layer single-character calibration models select values
present in the table but do not reliably select the requested key. At that point,
the research target became query-dependent lookup and multi-character copying.
The suite then had 18 passing tests, including a regression for shared-passage grouping. All five
development runs and the original failed outcomes are retained.

A subsequent read-only check found that the original four-character model gets
63/64 correct on both authorized and ungated **training** examples, versus 0/64
on unseen validation examples. That diagnosis was strong
overfitting, with the exact data/representation/optimization cause unresolved.
This distinction was missing from the initial qualification interpretation.

## Working local curriculum, 2026-09-10

The recovery now has a working 454,656-parameter byte Transformer. Fresh,
table-disjoint synthetic batches, calibrated initialization, and a curriculum
restore four-character lookup, authorization, and simultaneous natural-text
learning. The model starts with our random weights; subsequent stages continue
our own checkpoints. No pretrained model or tokenizer is imported.

The final mixed seed-17 model passed all three categories on 128 validation
tables and their reordered counterparts. An additional 512-table check scored
99.85% authorized answers (99.71% reordered), 100% withholding, and 100% ungated
retrieval. Every natural-text source beat its training-fitted byte unigram.
These are development checks on a fixed task family, not adversarial robustness
or a positive SCC result. The BPE variant remains unqualified.

See [the recovery report](../../../reports/RETRIEVAL_RECOVERY.md) for the curriculum,
replication, limitations, and complete experiment registry. At that milestone,
the suite had 34 passing tests, including exact curriculum restart and ancestry checks.
The runnable workflow is in [CORPUS_WORKFLOW.md](CORPUS_WORKFLOW.md).

## Coupling discovery, 2026-09-10

Exact unrolled attack gradients and intervention checkpointing are implemented
and validated. At that milestone, the suite had 43 passing tests. Three matched continuation
arms qualify cleanly, but the same 300-update removal plus 1,000-update recovery
procedure escapes all three while retaining measured capability. The escape-penalty
candidate discloses 511/512 forbidden answers on an additional development set;
authorized answers and retrieval both remain 512/512. Its continuation training
costs approximately 3.88 times ordinary continuation on this CPU.

All three arms also pass a policy-preserving benign uppercase edit. This is a
working experimental instrument and a rejected first candidate, not a mechanism
demonstration or reason to scale yet. The [report](../../../reports/COUPLING_DISCOVERY.md)
includes numerical convergence checks, all 23 run directories, attack-search
versus execution cost, and limitations. No GPU credits were spent.

## Stronger training attacks, 2026-09-10

The follow-up replaces three-step inner attacks with fresh copied-model AdamW
removal/recovery trajectories. The attack escapes the ordinary parent on all
three calibration seeds. Each non-control defender receives 16 full trajectories,
or 20,800 inner updates, with an explicitly first-order outer gradient. Matched
comparisons include ordinary continuation, refusal reinforcement, and coupling
weights 0.1 and 10. The suite has 46 passing tests, including native optimizer
agreement, gradient checks, and exact resumed training.

Both coupling models survive the language-retention threshold of one longer
attack, but a gentler 300-update attack escapes both models and both controls.
Those escapes persist on 128 new tables in original and reordered form and all
12,689 frozen natural-text validation blocks. All four models retain benign
editability. Approximately 73 times the ordinary continuation's forward token
positions did not increase the cheapest observed escape budget.

A separate experimental 300-update differentiable Adam probe fits in about
4.90 GB peak process RSS. Its derivative has numerical support only at extremely
small perturbations in this check; useful, stable outer optimization remains
unestablished. It was not substituted into the first-order campaign.

The [report](../../../reports/STRONG_ATTACK_RESULTS.md) and
[registry](../../../reports/STRONG_ATTACK_REGISTRY.md) retain all 27 run directories,
additional calibration, numerical checks, failures, and remaining avenues.
The [working standard](../../../WORKING_STANDARDS.md) governs claims and further
improvement. No GPU credits were spent; this remains a negative mechanism result.
