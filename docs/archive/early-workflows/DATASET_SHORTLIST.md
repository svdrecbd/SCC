> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# Existing datasets for the SCC pilot

Review date: 2026-09-09. The shortlist below preceded implementation. A bounded
Common Pile sample has since been downloaded, audited, and used in local baseline
checks; see [the qualification report](../../../reports/CORPUS_QUALIFICATION.md) and
[reproduction workflow](CORPUS_WORKFLOW.md). The final campaign recipe remains
unfrozen. No cloud training has been launched.

## Recommendation

Use an established ordinary-text corpus for the language component and generate
the protected authorization tasks ourselves. Our initialization, tokenizer
training, data selection, and optimization remain under our control. Creating
all ordinary-language text ourselves would add a separate research problem.

Audit a source-selected Common Pile sample first given the user's emphasis on
provenance, with FineWeb-Edu as the main alternative for a compact learning
comparison. Corpus suitability at approximately 50M parameters and a 1B-token
total training budget has not been established for either candidate. TinyStories
is a possible small-model calibration dataset, not the sole basis for a claim
about broad capability.

Final mixture proportions should follow inexpensive learnability checks. All
ordinary text and synthetic outer-training tokens count within the existing
per-model token budget. A larger candidate dataset does not imply a larger run.

## Candidates

### Common Pile v0.1, filtered sources / Comma training data

The project describes an 8TB raw collection of public-domain and openly licensed
text from 30 sources, with filtered data, collection code, and released training
mixtures. It evaluated the data by training 7B models on 1T and 2T tokens; that
evidence does not transfer automatically to our much smaller pilot.
[Paper](https://arxiv.org/abs/2506.05209).

The [Comma training dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)
consolidates modified filtered sources and documents source weights and repeated
exposures. We should select source families and record our own mixture rather
than copy a trillion-token recipe or ingest only the first files.

Start the audit with ordinary expository/educational text and books. Relevant
existing sources include [Wikimedia](https://huggingface.co/datasets/common-pile/wikimedia_filtered)
and [Project Gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered).
The cards explain their extraction and explicitly acknowledge possible licensing
metadata errors. The collection is not a blanket guarantee about every document.

The consolidated recipe includes a `data_provenance_initiative` source. Exclude
collections of ML task/instruction data from the initial language sample until
their contents can be screened against our evaluations. This is a proposed
exclusion rule, not a finding that a particular benchmark has leaked.

### FineWeb-Edu

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
is filtered Common Crawl text with a convenient `sample-10BT` configuration.
Its educational classifier was trained using Llama-3-70B-Instruct annotations:
the external model influences selection, while the source documents are web
text. This distinction must appear in our data provenance.

The card reports educational benchmark improvements in its experiments, along
with tradeoffs at stricter filtering thresholds. It lists ODC-By and Common
Crawl terms; that should not be represented as an individual open-content license
for every crawled page. [Card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md),
[Common Crawl terms](https://commoncrawl.org/terms-of-use).

This is a strong technical candidate to compare with the Common Pile sample.
Preserve document IDs, URLs, crawl IDs, and filtering scores. The published sample
size is measured in GPT-2 tokens; recount with our final tokenizer.

### TinyStories

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
offers existing synthetic stories and named train/validation resources. Its
card lists CDLA-Sharing-1.0 and identifies GPT-3.5/GPT-4 as the generators;
the GPT-4-only V2 is a distinct version. The published work demonstrates learning
in models smaller than our intended model, making it useful for calibration.
It has narrow linguistic coverage and an external teacher lineage. Do not use
stories alone to substantiate broad capability retention.
[Paper](https://arxiv.org/abs/2305.07759).

### SmolLM-Corpus: useful reference, additional complexity

[SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
combines an educational web subset, Cosmopedia v2 synthetic text, and Python-Edu.
Cosmopedia v2 uses Mixtral-generated text; Python-Edu requires additional content
retrieval and source-specific license handling. The collection is designed for
small language models, but we should not treat all components as one uniform
provenance or licensing regime. It is not needed to start the two-corpus audit.

## Actual intake observations

The public Hub API reported the following repositories as ungated and returned
these revisions on the review date. These are inspection snapshots, not final
training locks. Final files and manifests must be pinned and checksummed too.

| Repository | Inspected revision |
|---|---|
| HuggingFaceFW/fineweb-edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| common-pile/comma_v0.1_training_dataset | `5afc546db324e7f39f297ba757c9a60547151e7c` |
| common-pile/wikimedia_filtered | `0641bb84bd9b7162bcddf8be7836822161a9a342` |
| common-pile/project_gutenberg_filtered | `3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7` |
| common-pile/libretexts_filtered | `70388bca52b4a93515e14b1d56618fd7944988fd` |
| roneneldan/TinyStories | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |

Three-row previews from the public dataset viewer showed:

- FineWeb-Edu exposes text, document ID, URL, crawl/file information, language,
  token counts, and educational scores. Those preview rows did not carry an
  individual content-license field.
- Common Pile's Wikimedia subset exposes source, document ID, timestamps, text,
  and metadata including a license declaration, title, namespace, and provenance.
  The first three rows included a main page, a user page, and a chemistry cover
  page. This is a concrete reason to inspect namespaces and boilerplate even in
  a named filtered dataset. Three leading rows are not a quality-rate estimate.
- The Gutenberg preview exceeded our 2MB inspection limit and was not parsed.
  No document-quality conclusion is drawn from it.

These previews were live viewer responses, not frozen training artifacts. The
inspection does not establish corpus-wide contamination, quality, or license
error rates.

## What must be established before the final campaign

1. **Defined population.** Fix source families and a deterministic, source-aware
   sampling method. Audit samples across shards and document lengths; taking the
   first N documents can overrepresent one source. Preserve source metadata and
   filtering decisions. Remove unsuitable namespaces and malformed extraction.
2. **Leakage control.** Group documents/books/pages and near duplicates before
   splitting and chunking. Check across source families as well as within them.
   Keep known evaluation collections out of the training mixture and screen
   selected public benchmarks for overlap. Hash the final splits. State residual
   contamination uncertainty rather than claiming a perfect detector.
3. **Measured usefulness.** Use the same provisional tokenizer, architecture,
   compute allocation, and frozen development evaluations for a small comparison.
   Evaluate both candidates on the same held-out distributions and report results
   by source/domain. Comparing each only on its own validation set is not a fair
   quality comparison. Do not assume a 50M model will solve broad benchmarks.
4. **Correct units and implementation.** Train the tokenizer on training text
   only; record bytes, tokens, repeats, padding, and supervised tokens per source.
   Distinguish token mixture from example mixture. Efficient packing must preserve
   valid targets and the required authorization context.
5. **A frozen causal comparison.** Once the data recipe works, use identical
   outer-data schedules within each paired seed for invariant-only, early, and
   late coupling. Match auxiliary coupling dose between early and late treatment.
   The strongest final evaluation remains reserved from development.

These are experimental validity requirements, not a request for the user to
choose preprocessing heuristics. CPU inspection can precede any GPU comparison;
paid development must remain within the existing discovery allocation and the
workspace cap still needs reconciliation before cloud execution.

Hugging Face supports streaming and revision-specific loading, so inspection need
not download an entire multi-terabyte corpus. For final training, materialize and
hash the chosen sample instead of relying on a changing live stream.
[Streaming](https://huggingface.co/docs/datasets/stream),
[loading revisions](https://huggingface.co/docs/datasets/loading).
