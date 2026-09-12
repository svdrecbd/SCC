# Retrieval recovery — 2026-09-10

**The local foundation now works.** A 454,656-parameter Transformer learns
four-character lookup, the synthetic authorization rule, and ordinary-text
prediction together. Two model seeds pass the development gates. All weights
originate in our own random initialization; later curriculum stages continue
our own checkpoints. No pretrained model or tokenizer is used by this path.

This establishes a usable small baseline for developing SCC experiments. It is
not evidence that SCC coupling works, survives white-box modification, or scales
to a generally capable language model. No GiveMeANode job or GPU credit was used.

The passed gate is the local **baseline learnability gate**. It does not complete
Phase 0 or Tier 0 of the original roadmap, whose exit conditions also require
end-to-end attack evaluation. The exploratory recovery protocol is not a
preregistered confirmatory study, and retaining all trials does not remove
selection effects from repeated development validation.

## What was failing, and what changed

The original BPE model obtained 63/64 useful answers on training cases but 0/64
on validation, in both authorized and ungated retrieval. That is strong evidence
of overfitting. Natural-text loss nevertheless improved. The original outcomes
remain in [the qualification report](CORPUS_QUALIFICATION.md).

Recovery changed three things: fresh procedurally generated task batches with
all four counterfactual queries; the small model's initialization scale; and
the order in which it learns the task. Table identity now excludes query, order,
and permission, so a whole mapping belongs to one split.

The most informative controlled comparison used identical one-character atomic
tasks, model/data seeds, optimizer, and training schedule. Initialization std
0.02 gave 24.2% validation retrieval; std 0.05 or 0.10 gave 100%, including
reordering. Std 0.05 also passed with model seed 23. This supports an
initialization effect in this small control. It does not identify the full
optimization mechanism or prescribe initialization at larger model widths.

Four-character tasks still failed when started independently. Continuing our
successful single-character lookup model recovered four-character retrieval.
Continuing that model on the original readable prompts recovered authorization.
The final mixed stage retained these abilities while learning natural text.
These changes jointly establish a working procedure; they are not a factorial
causal decomposition of the original failure.

The BPE branch remains unqualified, including its deeper, longer calibration.
The successful foundation therefore uses byte tokens. It does not substitute
single-character success for the original four-character task, and it does not
claim that changing Python or replacing Common Pile solved the task.

## Retained training sequence

The backbone is a conventional dense causal decoder: two pre-norm layers,
width 128, four attention heads, GELU, learned absolute positions, tied
embeddings, context 192, and 260 byte/control token IDs. Attention is global
and causal. The successful configuration uses no rotary/ALiBi positions, local
attention restriction, auxiliary echo objective, answer parser in inference,
hardcoded lookup operation, or output filter.

All stages use batch 32, AdamW at peak learning rate 0.001 with warmup/decay,
fp32 CPU execution, and initialization std 0.05. Residual projections receive
the documented depth scaling. Each transition loads our own weights and resets
the optimizer; it is a curriculum transition, not an exact optimizer resume.

| Stage | Training | Updates | Seed-17 run |
|---|---|---:|---|
| 1 | Atomic one-character lookup, random initialization | 4,000 | [01w](../runs/online-01w-init005/result.json) |
| 2 | Atomic four-character lookup, from stage 1 | 4,000 | [02g](../runs/online-02g-lookup-curriculum/result.json) |
| 3 | Original readable four-character prompts; retrieval and paired permissions | 4,000 | [04c](../runs/online-04c-original-byte-curriculum/result.json) |
| 4 | Same tasks mixed with all four natural-text sources | 8,000 | [06](../runs/online-06-byte-mixed/result.json) |

Model seed 23 repeats this sequence with identical data seeds and schedules:
[stage 1](../runs/online-01z-seed23/result.json),
[stage 2](../runs/online-07a-seed23-four/result.json),
[stage 3](../runs/online-07b-seed23-permission/result.json), and
[stage 4](../runs/online-07c-seed23-mixed/result.json).
This is a model-initialization replication, not two independently sampled corpora.

Stages 1–3 use data seed 101 and intentionally revisit some training mappings
across curriculum stages. Stage 4 uses seed 303. Each stage samples tables
without replacement; the complete lineage contains 93,272 distinct training
tables per model. The final verifier traverses all ancestors and deduplicates
revisits. Early run-local counts must not be summed as independent tables.

The final stage's batch probabilities are 30% ungated retrieval, 30% paired
permissions, and 10% each Wikipedia, Pressbooks, LibreTexts, and Gutenberg.
Task loss supervises answers only; text loss supervises next-token prediction.
Measured natural text accounts for 18,893,928 of 30,952,784 input tokens (61.0%)
in each final run. These are bytes/control tokens, not BPE tokens. The audited
documents are the same frozen 3,313-document, 23.99 MB sample used previously;
old fixed synthetic examples in the prepared file are not sampled for training.

## Behavioral qualification

The protocol was recorded before these recovery runs: at least 95% exact
answers and 90% tables with every queried key correct, separately in each
category and both orderings. Correctness requires EOS termination. Always
refusing fails the permitted-answer gate. The independent label parser is used
for data checking, never to answer on behalf of the model.

Both final models scored 512/512 in every category on the main 128-table
validation set and again after cyclic reordering. Additional validation used
512 tables from seed 17419, with four queries per category:

| Model seed | Category | Additional original order | Additional reordered | Worst fraction of tables with all four queries correct |
|---:|---|---:|---:|---:|
| 17 | Authorized answer | 2,045 / 2,048 (99.85%) | 2,042 / 2,048 (99.71%) | 98.83% |
| 17 | Withholding | 2,048 / 2,048 (100%) | 2,048 / 2,048 (100%) | 100% |
| 17 | Ungated retrieval | 2,048 / 2,048 (100%) | 2,048 / 2,048 (100%) | 100% |
| 23 | Authorized answer | 2,048 / 2,048 (100%) | 2,048 / 2,048 (100%) | 100% |
| 23 | Withholding | 2,048 / 2,048 (100%) | 2,048 / 2,048 (100%) | 100% |
| 23 | Ungated retrieval | 2,048 / 2,048 (100%) | 2,047 / 2,048 (99.95%) | 99.80% |

The remaining errors include false withholding, truncated strings, and wrong
characters. They are preserved in the raw predictions. No further fitting to
these additional cases was performed. There were no observed unauthorized
disclosures in these checks; that is an observation on this task distribution,
not a robustness guarantee. Queries and reordered cases share tables and are
not all independent statistical observations.

Evidence: [seed 17 additional predictions](../artifacts/retrieval-recovery/mixed-additional-validation.json),
[seed 23 additional predictions](../artifacts/retrieval-recovery/mixed-seed23-additional-validation.json).
Training/validation separation is checked across the entire checkpoint ancestry.
The final 10% test partition remains unused. All reported evaluations are
development validation, not a sealed final test after an unseen experimental design.

## Natural-text qualification

Each source is evaluated on 64 fixed held-out blocks. The comparison is an
add-one-smoothed unigram fitted separately to each source's training data.
Lower NLL is better; units are nats per supervised byte/control token.

| Source | Training unigram | Mixed seed 17 | Mixed seed 23 |
|---|---:|---:|---:|
| Wikipedia | 3.289 | 1.900 | 1.946 |
| Pressbooks | 3.217 | 1.683 | 1.724 |
| LibreTexts | 3.243 | 1.773 | 1.804 |
| Gutenberg | 3.169 | 1.874 | 1.894 |

Both models improve on every source, and on their own pre-mix losses.
This supports contextual text prediction alongside task competence. It does
not establish broad reasoning, useful prose generation, representative corpus
coverage, or comparable quality to a larger BPE model. NLL values from the
earlier BPE report cannot be compared directly with these byte values.

The original audit's limitations remain: bounded shard prefixes, five
validation Gutenberg books, extraction artifacts, source-selection bias, and
declared rather than independently proven licenses. A billion-token campaign
requires a larger audited corpus, not repeated replay of this small pool.

## Audit trail and reproducibility

The [full registry](RETRIEVAL_RECOVERY_REGISTRY.md) includes all 43 online
development runs: 9 actual lookup-gate passes, 33 failed runs, and one query-echo
diagnostic that is explicitly excluded from lookup success. The five original
qualification runs are preserved separately. Failed variants include alternative
positions, depth, learning rates, output tying, batch shuffling, key vocabularies,
longer schedules, and an unsuccessful echo curriculum. None are silently discarded.

Each run preserves its configuration, exact source/dependency snapshot, raw
predictions, table stream, source exposure, model, optimizer, and RNG state.
The 34-test suite checks causality, data/label/split contracts, corruption,
tokenization, exact uninterrupted-versus-resumed CPU training (including dropout,
shuffle buffers, and curriculum initialization), and ancestor integrity.

Final combined verification checks source snapshots, full checkpoint ancestry,
split separation, behavioral gates, and per-source language baselines:
[seed 17](../artifacts/retrieval-recovery/foundation-verification-seed17.json),
[seed 23](../artifacts/retrieval-recovery/foundation-verification-seed23.json).
The seed-17 winning sequence logged 5.29 minutes inside training steps. Across
all 43 online attempts the sum was 37.66 minutes. These exclude evaluation,
setup, checkpoint writes, and engineering work; they are not elapsed project
time or GPU throughput estimates. Cloud expenditure for this work was $0.

Use [the runnable workflow](../docs/archive/early-workflows/CORPUS_WORKFLOW.md) to reproduce the entire
curriculum from random weights. Exact restart requires the saved configuration,
source, dependencies, and environment; moving between software versions or
devices is not promised to be bitwise identical.

## What this permits next

The immediate learnability blockage has been cleared for this byte-model
foundation. The next scientific step is to implement the proposed coupling
objective and compare it with matched uncoupled controls under bounded attacks.
Those controls must share the curriculum, exposure, initialization policy, and
data schedule. This work does not yet support claims of tamper resistance.

Before paid scale-up, the model/tokenizer choice, wider task families, unseen
identities and table sizes, independent capability measures, larger corpus,
CUDA throughput, and remote checkpoint recovery still need qualification.
The successful baseline makes those experiments concrete; it does not replace them.
