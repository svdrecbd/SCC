# Corpus qualification — 2026-09-09

A reproducible development corpus and training pipeline are implemented and tested. Natural-text learning passed its development comparison. Useful authorization and query-dependent retrieval remain unqualified in these original runs. This historical report covers the baseline and its initial diagnostics, not SCC or the final 50M campaign. The later [retrieval recovery report](RETRIEVAL_RECOVERY.md) documents the working byte-model curriculum.

## Corpus and audit

The selected natural-text sample contains **3,313 documents** (23.99 MB of UTF-8 text), plus **17,165** generated task examples.

| Source | Train documents | Validation documents | Test documents | Selected MB |
|---|---:|---:|---:|---:|
| wikimedia | 1,096 | 146 | 134 | 6.00 |
| pressbooks | 992 | 126 | 151 | 6.00 |
| libretexts | 465 | 51 | 88 | 6.00 |
| gutenberg | 52 | 5 | 7 | 5.99 |

The audit preserves upstream IDs, revisions, declared licenses, book identities, and file receipts. Textbook chapters stay with their books; detected shared passages are grouped before splitting. The independent identity/content/shared-passage checks found 0 cross-split violations within their stated detection scope.

The first audit was retained. Manual review found a vandalized Wikibooks chemistry page, so the revised encyclopedia component keeps Wikipedia only. The initial book sample had one validation book; selecting complete shorter works within the same byte cap increased this to five. These changes preceded model qualification.

The four-source acquisition scanned 30,300 records from seeded shards and bounded prefixes. It is not a population-representative sample of Common Pile. Older short works, textbook extraction artifacts, missing figures, and source-selection bias remain. License declarations were checked against an allowlist, not independently proven.

## Tokenizer and reproducibility

A **4,096-token byte-level BPE tokenizer** was trained from scratch using only training records. No pretrained tokenizer or model weights were imported. Rebuilding from the frozen acquired records produced:

- audited records identical: **yes**.
- recipe records identical: **yes**.
- tokenizer identical: **yes**.

Eighteen automated tests cover model causality, learning, masking, corruption checks, tokenizer training-split isolation, and exact CPU restart including grouped sampling and exposure counters. Each run stores its configuration, source snapshot, optimizer, RNGs, sampler state, and checkpoints.

Current development uses an isolated uv environment with Python 3.14.7, PyTorch 2.14.0, and locked transitive dependencies. The first two mixed runs used Python 3.12.7 / PyTorch 2.12.0. The newer environment passes the restart tests and loads the older checkpoint with identical greedy outputs on 32 sampled prompts. Cross-version training equivalence is not claimed.

## Learning result

One randomly initialized **945,664-parameter** Transformer trained for **20,000 updates**, consuming **30,496,754 input tokens** and **23,979,866 supervised tokens**. Measured training-step time was 9.38 CPU minutes; this excludes setup, checkpoint writes, and evaluation. No GiveMeANode compute was used.

The comparison uses fixed validation blocks and an add-one-smoothed unigram fitted separately on each training source. Lower negative log likelihood (NLL, nats per BPE token) is better. This tests contextual prediction; it does not establish general reasoning.

| Source | Random initialization NLL | Source unigram NLL | Trained NLL |
|---|---:|---:|---:|
| wikimedia | 8.330 | 6.842 | 4.785 |
| pressbooks | 8.336 | 6.925 | 4.611 |
| libretexts | 8.340 | 7.114 | 4.709 |
| gutenberg | 8.336 | 6.682 | 4.933 |

The baseline beats the unigram on all four selected text sources.

| Synthetic validation behavior | Correct / evaluated | Exact match |
|---|---:|---:|
| authorized | 0 / 64 | 0.0% |
| unauthorized | 64 / 64 | 100.0% |
| retrieval | 0 / 64 | 0.0% |
| addition | 7 / 64 | 10.9% |

Unauthorized exact match means correctly withholding; the other rows require the correct useful answer. Always withholding is not a successful control. The first 2,500-step run scored 0/64 on authorized retrieval and ungated retrieval despite learning language and withholding. That failure prompted the longer run with the same data recipe; both runs and their configurations are retained. These are exploratory development results from one model seed and IID task instances.

**The original four-character mixed baseline fails the task-competence gate.** Lower text loss and perfect withholding cannot compensate for failure to return permitted answers.

## Development diagnostics — all trials

After the original mixed run failed, a deeper model was trained on authorization and retrieval alone. We then introduced explicitly labeled single-character values to separate lookup/permission learning from multi-character copying. Calibration runs reuse the frozen original tokenizer and the same natural documents; their generated task instances and splits differ. Both task recipes passed independent leakage checks. Calibration success would not establish competence on the original four-character task.

| Run | Parameters | Updates | Value characters | Authorized | Withhold | Ungated retrieval | Addition |
|---|---:|---:|---:|---:|---:|---:|---:|
| [corpus-qualification-v1](../../../runs/corpus-qualification-v1/qualification.json) | 945,664 | 2,500 | 4 | 0.0% | 100.0% | 0.0% | 1.6% |
| [corpus-qualification-v2](../../../runs/corpus-qualification-v2/qualification.json) | 945,664 | 20,000 | 4 | 0.0% | 100.0% | 0.0% | 10.9% |
| [task-diagnostic-v1](../../../runs/task-diagnostic-v1/qualification.json) | 1,342,208 | 8,000 | 4 | 0.0% | 100.0% | 0.0% | 0.0% |
| [corpus-calibration-v1](../../../runs/corpus-calibration-v1/qualification.json) | 945,664 | 10,000 | 1 | 26.6% | 100.0% | 29.7% | 1.6% |
| [corpus-calibration-v2](../../../runs/corpus-calibration-v2/qualification.json) | 1,342,208 | 10,000 | 1 | 25.0% | 100.0% | 28.1% | 7.8% |

### Follow-up: training versus unseen examples

Read-only scoring after the initial report separates task fitting from generalization. Each cell below uses 64 examples per category. No weights were updated and the test split stayed unused.

| Run | Authorized train | Authorized validation | Retrieval train | Retrieval validation |
|---|---:|---:|---:|---:|
| corpus-calibration-v1 | 59.4% | 26.6% | 53.1% | 29.7% |
| corpus-calibration-v2 | 65.6% | 25.0% | 56.2% | 28.1% |
| corpus-qualification-v2 | 98.4% | 0.0% | 98.4% | 0.0% |

The original four-character baseline gets 63/64 correct in each training category and 0/64 on unseen validation examples. This is strong evidence of overfitting: it can fit the training tasks, but the learned behavior does not transfer to fresh tables. The initial interpretation did not distinguish fitting from generalization and was incomplete. The exact cause remains unresolved; finite repeated synthetic instances, representation, optimization, and capacity still need controlled comparisons.

[Training/validation diagnostic and predictions](../../../artifacts/corpus-qualification/train-vs-validation-diagnostic.json)

Every behavioral cell uses 64 development validation examples; results across these different tasks are not directly interchangeable. The task-only diagnostic did not train addition or natural-text prediction. A roughly 25–30% single-character retrieval score can be achieved by selecting an arbitrary table value; it is not evidence of reliable query-dependent lookup. Both calibration models returned a value present in the table on all 64 authorized and all 64 ungated examples, but correct-key selection remained weak. Always returning the first table value scores 20/64 authorized and 19/64 ungated on this same sample, matching or exceeding both models. This localizes the next development target to query-dependent selection; it does not establish its underlying optimization or representation cause. No seed sweep or claim-bearing final test was run.

The five experiments used 19.05 minutes of measured CPU training-step time in total. This excludes preparation, evaluation, and checkpoint writes. GiveMeANode spending remains **$0**.

## Actual mixture and limits

Natural text accounted for **76.6%** of the longer run's input-token exposure. The configuration specifies batch-source probabilities, not token percentages. Per-source input, supervised-token, and example counters are stored in the run result. Early/late/SCC arms have not been run.

The held-out test split was prepared and checked for leakage but was not used for model evaluation or tokenizer training. The current screen checks benchmark names, not arbitrary benchmark-answer overlap. Stronger structural task tests and held-out attacks remain necessary. The corpus contains about 7.55 million stored supervised tokens across all splits; repeatedly cycling this sample is not a substitute for preparing the larger final training corpus.

The next funded-scale gate remains a timed approximately 50M baseline with adequate task competence, an expanded audited data population, and a frozen evaluation protocol. These local results do not establish that gate, SCC effectiveness, broad model capability, or robustness to modification.

A final code review strengthened passage grouping to compare every earlier document sharing an anchor. A regression test covers a case the original grouping could miss. On the acquired population, the stronger matcher retains the identical documents and moves four documents between splits after regrouping. The original frozen recipes independently pass the cross-split checker and all reported runs use those recipes. The revised audit is retained separately; reproducing the original record bytes requires the saved original source. The all-pairs matcher is a bounded reference implementation, not a corpus-scale deduplication index.

## Inspect and reproduce

- [Audit](../../../artifacts/corpus-qualification/audited-v2/audit.json)
- [Leakage checks](../../../artifacts/corpus-qualification/leakage.json)
- [Calibration leakage checks](../../../artifacts/corpus-qualification/calibration-leakage.json)
- [Recipe manifest](../../../artifacts/corpus-qualification/recipe/recipe.json)
- [Tokenizer manifest](../../../artifacts/corpus-qualification/tokenizer/tokenizer_manifest.json)
- [Rebuild verification](../../../artifacts/corpus-qualification/rebuild_verification.json)
- [Revised passage-grouping audit](../../../artifacts/corpus-qualification/audited-all-pairs/audit.json)
- [Grouping change comparison](../../../artifacts/corpus-qualification/all-pairs-verification.json)
- [Revised audit leakage checks](../../../artifacts/corpus-qualification/all-pairs-leakage.json)
- [Test verification](../../../artifacts/corpus-qualification/test-verification.json)
- [Retrieval diagnostic](../../../artifacts/corpus-qualification/retrieval-diagnostic.json)
- [Current environment](../../../artifacts/corpus-qualification/environment-uv.json)
- [Cross-version checkpoint check](../../../artifacts/corpus-qualification/environment-portability.json)
- [Full evaluation and predictions](../../../runs/corpus-qualification-v2/qualification.json)
- [Training result and source exposure](../../../runs/corpus-qualification-v2/training/result.json)
- [Saved source manifest](../../../runs/corpus-qualification-v2/training/source/source_manifest.json)
