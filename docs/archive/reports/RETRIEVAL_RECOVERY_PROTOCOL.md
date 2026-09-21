# Retrieval recovery: development protocol — 2026-09-10

The previous four-character model scored 63/64 on training examples and 0/64
on validation examples in both authorized and ungated retrieval. This protocol
addresses that generalization failure before SCC training or paid GPU use.

## Data and gates

- Every training table is newly sampled without replacement within a run.
- All four keys are queried for every table. Values are distinct, so ignoring
  the query cannot solve the table's complete set of queries.
- Permission batches contain both authorized and unauthorized versions of every
  query on the same table. Category labels and answers are excluded from input.
- The table's key/value mapping alone defines its identity and split. Query,
  ordering, user identity, renderer, and permission do not affect the split.
- Training accepts only the 80% training hash partition. The 10% validation
  partition supplies development checks; the final 10% test partition is unused.
- An independent parser recomputes every training answer from its displayed
  premise. The model receives tokens and completion loss; the parser does not
  provide predictions or gate model outputs.
- Main evaluation uses 128 fixed validation tables, seed 731, with four queries
  per category and a second evaluation after cyclically reordering each table.
- Each trained category must reach at least 95% exact answers and at least 90%
  tables with all four queries correct, on both orderings. EOS is required.
- Periodic evaluation uses eight of these tables for progress only. These are
  development gates, not a preregistered SCC result or independent final test.
- Successful configurations should be checked on additional tables and model
  seeds. All failed configurations and source snapshots remain available.

## Planned progression

The original plan started each main stage independently from random initialization.
The successful route below instead uses a curriculum of our own checkpoints.
The first baseline uses two
layers, width 128, four heads, context 192, and a byte vocabulary. Default runs
use 4,000 AdamW updates, batch size 32, learning rate 0.001 with warmup/decay,
and model/data seeds 17/101. Changes beyond those below must be recorded.

1. Compact renderer, one-character values, retrieval only.
2. Increase values to four characters.
3. Add paired permission examples alongside ungated retrieval.
4. Restore the original human-readable task prompts with byte encoding.
5. Use the frozen own-trained BPE tokenizer on those original prompts.
6. Restore the four Common Pile components in mixed training and check both
   task generalization and held-out language loss.

The initial changes jointly alter data freshness, query coverage, encoding, and
task difficulty. Success would establish a working path, not isolate the causal
contribution of every change. Adjacent stages narrow particular comparisons.
The earlier addition task is outside this retrieval diagnosis.

## Evidence and limits

Configurations live in `configs/online/`; runs retain exact configurations,
source/dependency snapshots, per-source token exposure, table-stream state,
optimizer/RNG state, evaluation records, raw predictions, and checkpoints.
The source files are `scc/online_tasks.py` and `scc/online_train.py`.

The task still uses four known keys and four known user names. Reordering is a
limited structural check. Unseen key vocabularies, larger tables, novel prompt
families, adversarial modifications, broad capability, and SCC coupling require
later experiments. No cloud job is part of this local protocol.

## Recorded development deviations

The compact one-character baseline failed. Development controls subsequently
tested four layers, query-last and atomic renderings, rotary and ALiBi positions,
larger learning rates, untied embeddings, shuffled batches, 12,000 updates,
variable keys, one attention head, reduced EOS loss weight, and a local first
attention layer. None passed the actual lookup gate at initialization std 0.02.
These are separate controls, not an exhaustive factorial comparison.

A diagnostic that merely echoes the query reached 100%. Transferring its own
weights to lookup did not recover retrieval. The echo run is **not** a successful
retrieval model even though its historical result uses the same category label.
This exploratory transfer is the exception to independent initialization; its
parent checkpoint, hash, and optimizer reset are recorded in the run contract.

Changing the initial standard deviation from 0.02 to 0.05 or 0.10 recovered atomic
one-character lookup. The matched current-code 0.02 control remained at 24.2%;
0.05 reached 100% for model seeds 17 and 23, including reordered validation.
All use two layers, width 128, four globally causal attention heads, learned
absolute positions, tied embeddings, and ordinary completion cross-entropy.
No parser, lookup primitive, frozen model weights, or output filter is in the
model. The initializer affects all embeddings and linear weights, with residual
projections additionally divided by sqrt(2 * layers). The result supports a
configuration effect in this small control; its exact optimization mechanism
and suitability at larger widths have not been established.

Four-character retrieval from random initialization still failed. Continuing
the successful atomic one-character model on atomic four-character tables
recovered 100% validation retrieval. Continuing that model on original readable
prompts and paired authorization examples recovered all three categories.
An intermediate compact-permission branch failed and was not used as a parent.
The successful original-format model then initialized mixed natural-text training.

The retained path is therefore 4,000 atomic one-character updates, 4,000 atomic
four-character updates, 4,000 original-format retrieval/permission updates, and
8,000 mixed updates. Every transition loads our own weights and resets AdamW;
initialization std is 0.05 throughout. Model seeds 17 and 23 follow this same
schedule. The first three stages use data seed 101, intentionally revisiting
some tables across stages. The final mixed stage uses data seed 303. Tables are
unique within each stage; lineage-wide unique counts must deduplicate revisits.
All ancestors are restricted to the training partition.

The final mix uses batch probabilities 0.30 retrieval, 0.30 paired permissions,
and 0.10 for each natural-text source. It uses the frozen audited documents,
re-encoded as bytes. Ordinary text supervises next-token prediction; task batches
supervise completion tokens only. Per-source token exposure is measured.

The BPE branch did not qualify, including a deeper, longer one-character run.
It was not used in the successful lineage. Byte encoding is the retained local
foundation; suitability of a learned tokenizer at pilot scale remains open.
This is a documented change from the original progression, not a claim that
every proposed stage succeeded.

Each retained stage passes the original behavioral thresholds. Final natural
text must also beat the source-specific training unigram and pre-mix model on
the same held-out blocks. Additional validation uses 512 tables, seed 17419,
with both orderings. These are exploratory development checks. All intermediate
failures remain in the registry; the final test partition is still reserved.
