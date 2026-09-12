"""Rebuild the development qualification report from saved artifacts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, default=Path("configs/qualification_experiments.json"))
    args = parser.parse_args()
    corpus, run = args.corpus.resolve(), args.run.resolve()
    audit = json.loads((corpus / "audited-v2/audit.json").read_text())
    recipe = json.loads((corpus / "recipe/recipe.json").read_text())
    leakage = json.loads((corpus / "leakage.json").read_text())
    rebuild = json.loads((corpus / "rebuild_verification.json").read_text())
    prepared = json.loads((corpus / "prepared/manifest.json").read_text())
    result = json.loads((run / "qualification.json").read_text())
    training = result["training"]
    sources = ["wikimedia", "pressbooks", "libretexts", "gutenberg"]
    tasks = ["authorized", "unauthorized", "retrieval", "addition"]
    natural_total = sum(audit["selection"][name]["selected"] for name in sources)
    natural_bytes = sum(audit["selection"][name]["utf8_bytes"] for name in sources)
    natural_input = sum(training["exposure"][name]["input_tokens"] for name in sources)
    language_pass = all(result["by_group"][name]["beats_unigram"] for name in sources)
    lines = ["# Corpus qualification — 2026-09-09", "",
             "A reproducible development corpus and training pipeline are implemented and tested. "
             "Natural-text learning passed its development comparison. Useful authorization and query-dependent retrieval "
             "remain unqualified in these original runs. This historical report covers the baseline and its initial diagnostics, not SCC or the final 50M campaign. "
             "The later [retrieval recovery report](RETRIEVAL_RECOVERY.md) documents the working byte-model curriculum.", "",
             "## Corpus and audit", "",
             f"The selected natural-text sample contains **{natural_total:,} documents** "
             f"({natural_bytes / 1e6:.2f} MB of UTF-8 text), plus **{sum(recipe['record_counts'][name] for name in tasks):,}** generated task examples.", "",
             "| Source | Train documents | Validation documents | Test documents | Selected MB |",
             "|---|---:|---:|---:|---:|"]
    for name in sources:
        counts = audit["split_documents"][name]
        lines.append(f"| {name} | {counts.get('train', 0):,} | {counts.get('validation', 0):,} | {counts.get('test', 0):,} | {audit['selection'][name]['utf8_bytes']/1e6:.2f} |")
    lines += ["", "The audit preserves upstream IDs, revisions, declared licenses, book identities, and file receipts. "
              "Textbook chapters stay with their books; detected shared passages are grouped before splitting. "
              "The independent identity/content/shared-passage checks found "
              f"{sum(leakage['checks'].values())} cross-split violations within their stated detection scope.", "",
              "The first audit was retained. Manual review found a vandalized Wikibooks chemistry page, so the revised "
              "encyclopedia component keeps Wikipedia only. The initial book sample had one validation book; selecting "
              "complete shorter works within the same byte cap increased this to five. These changes preceded model qualification.", "",
              "The four-source acquisition scanned 30,300 records from seeded shards and bounded prefixes. It is not a "
              "population-representative sample of Common Pile. Older short works, textbook extraction artifacts, missing "
              "figures, and source-selection bias remain. License declarations were checked against an allowlist, not independently proven.", "",
              "## Tokenizer and reproducibility", "",
              f"A **{prepared['tokenizer']['vocab_size']:,}-token byte-level BPE tokenizer** was trained from scratch using only training records. "
              "No pretrained tokenizer or model weights were imported. Rebuilding from the frozen acquired records produced:", ""]
    for name, passed in rebuild.items():
        lines.append(f"- {name.replace('_', ' ')}: **{'yes' if passed else 'NO'}**.")
    lines += ["", "Eighteen automated tests cover model causality, learning, masking, corruption checks, tokenizer "
              "training-split isolation, and exact CPU restart including grouped sampling and exposure counters. "
              "Each run stores its configuration, source snapshot, optimizer, RNGs, sampler state, and checkpoints.", "",
              "Current development uses an isolated uv environment with Python 3.14.7, PyTorch 2.14.0, "
              "and locked transitive dependencies. The first two mixed runs used Python 3.12.7 / PyTorch 2.12.0. "
              "The newer environment passes the restart tests and loads the older checkpoint with identical greedy outputs "
              "on 32 sampled prompts. Cross-version training equivalence is not claimed.", "",
              "## Learning result", "",
              f"One randomly initialized **{training['parameter_count']:,}-parameter** Transformer trained for "
              f"**{training['completed_steps']:,} updates**, consuming **{training['input_tokens']:,} input tokens** "
              f"and **{training['loss_tokens']:,} supervised tokens**. Measured training-step time was "
              f"{training['compute_seconds']/60:.2f} CPU minutes; this excludes setup, checkpoint writes, and evaluation. "
              "No GiveMeANode compute was used.", "",
              "The comparison uses fixed validation blocks and an add-one-smoothed unigram fitted separately on each training source. "
              "Lower negative log likelihood (NLL, nats per BPE token) is better. This tests contextual prediction; it does not establish general reasoning.", "",
              "| Source | Random initialization NLL | Source unigram NLL | Trained NLL |",
              "|---|---:|---:|---:|"]
    for name in sources:
        value = result["by_group"][name]
        lines.append(f"| {name} | {value['initial_nll']:.3f} | {value['unigram_nll']:.3f} | {value['final_nll']:.3f} |")
    lines += ["", f"The baseline {'beats' if language_pass else 'does not beat'} the unigram on all four selected text sources.", "",
              "| Synthetic validation behavior | Correct / evaluated | Exact match |",
              "|---|---:|---:|"]
    for name in tasks:
        score = result["behavior"][name]
        lines.append(f"| {name} | {round(score['exact_match'] * score['examples'])} / {score['examples']} | {100*score['exact_match']:.1f}% |")
    lines += ["", "Unauthorized exact match means correctly withholding; the other rows require the correct useful answer. "
              "Always withholding is not a successful control. The first 2,500-step run scored 0/64 on authorized retrieval "
              "and ungated retrieval despite learning language and withholding. That failure prompted the longer run with the same data recipe; "
              "both runs and their configurations are retained. These are exploratory development results from one model seed and IID task instances.", "",
              "**The original four-character mixed baseline fails the task-competence gate.** Lower text loss and "
              "perfect withholding cannot compensate for failure to return permitted answers.", "",
              "## Development diagnostics — all trials", "",
              "After the original mixed run failed, a deeper model was trained on authorization and retrieval alone. "
              "We then introduced explicitly labeled single-character values to separate lookup/permission learning from "
              "multi-character copying. Calibration runs reuse the frozen original tokenizer and the same natural documents; "
              "their generated task instances and splits differ. Both task recipes passed independent leakage checks. "
              "Calibration success would not establish competence on the original four-character task.", "",
              "| Run | Parameters | Updates | Value characters | Authorized | Withhold | Ungated retrieval | Addition |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    total_cpu = 0.0
    for spec in json.loads(args.experiments.read_text()):
        path = Path(spec["run"]).resolve()
        trial = json.loads((path / "qualification.json").read_text())
        tr = trial["training"]
        total_cpu += tr["compute_seconds"]
        scores = [f"{100*trial['behavior'][name]['exact_match']:.1f}%" for name in tasks]
        lines.append(f"| [{path.name}]({path / 'qualification.json'}) | {tr['parameter_count']:,} | "
                     f"{tr['completed_steps']:,} | {spec['value_length']} | " + " | ".join(scores) + " |")
    generalization_path = corpus / "train-vs-validation-diagnostic.json"
    if generalization_path.exists():
        diagnosis = json.loads(generalization_path.read_text())
        lines += ["", "### Follow-up: training versus unseen examples", "",
                  "Read-only scoring after the initial report separates task fitting from generalization. "
                  "Each cell below uses 64 examples per category. No weights were updated and the test split stayed unused.", "",
                  "| Run | Authorized train | Authorized validation | Retrieval train | Retrieval validation |",
                  "|---|---:|---:|---:|---:|"]
        for name, value in diagnosis["models"].items():
            scores = [value[split][category]["exact_match"]
                      for category in ("authorized", "retrieval")
                      for split in ("train_behavior", "validation_behavior")]
            lines.append(f"| {name} | " + " | ".join(f"{score*100:.1f}%" for score in scores) + " |")
        lines += ["", "The original four-character baseline gets 63/64 correct in each training category and 0/64 on "
                  "unseen validation examples. This is strong evidence of overfitting: it can fit the training tasks, "
                  "but the learned behavior does not transfer to fresh tables. The initial interpretation did not distinguish "
                  "fitting from generalization and was incomplete. The exact cause remains unresolved; finite repeated "
                  "synthetic instances, representation, optimization, and capacity still need controlled comparisons.", "",
                  f"[Training/validation diagnostic and predictions]({generalization_path})"]
    lines += ["", "Every behavioral cell uses 64 development validation examples; results across these different tasks "
              "are not directly interchangeable. The task-only diagnostic did not train addition or natural-text prediction. "
              "A roughly 25–30% single-character retrieval score can be achieved by selecting an arbitrary table value; "
              "it is not evidence of reliable query-dependent lookup. Both calibration models returned a value present "
              "in the table on all 64 authorized and all 64 ungated examples, but correct-key selection remained weak. "
              "Always returning the first table value scores 20/64 authorized and 19/64 ungated on this same sample, "
              "matching or exceeding both models. This localizes the next development target to query-dependent selection; "
              "it does not establish its underlying optimization or representation cause. No seed sweep or claim-bearing final test was run.", "",
              f"The five experiments used {total_cpu/60:.2f} minutes of measured CPU training-step time in total. "
              "This excludes preparation, evaluation, and checkpoint writes. GiveMeANode spending remains **$0**.", "",
              "## Actual mixture and limits", "",
              f"Natural text accounted for **{100*natural_input/training['input_tokens']:.1f}%** of the longer run's input-token exposure. "
              "The configuration specifies batch-source probabilities, not token percentages. Per-source input, supervised-token, "
              "and example counters are stored in the run result. Early/late/SCC arms have not been run.", "",
              "The held-out test split was prepared and checked for leakage but was not used for model evaluation or tokenizer training. "
              "The current screen checks benchmark names, not arbitrary benchmark-answer overlap. Stronger structural task tests and "
              "held-out attacks remain necessary. The corpus contains about "
              f"{sum(value['loss_tokens'] for value in prepared['split_counts'].values())/1e6:.2f} million stored supervised tokens "
              "across all splits; repeatedly cycling this sample is not a substitute for preparing the larger final training corpus.", "",
              "The next funded-scale gate remains a timed approximately 50M baseline with adequate task competence, "
              "an expanded audited data population, and a frozen evaluation protocol. These local results do not establish that gate, "
              "SCC effectiveness, broad model capability, or robustness to modification.", "",
              "A final code review strengthened passage grouping to compare every earlier document sharing an anchor. "
              "A regression test covers a case the original grouping could miss. On the acquired population, the stronger "
              "matcher retains the identical documents and moves four documents between splits after regrouping. "
              "The original frozen recipes independently pass the cross-split checker and all reported runs use those recipes. "
              "The revised audit is retained separately; reproducing the original record bytes requires the saved original source. "
              "The all-pairs matcher is a bounded reference implementation, not a corpus-scale deduplication index.", "",
              "## Inspect and reproduce", "",
              f"- [Audit]({corpus / 'audited-v2/audit.json'})",
              f"- [Leakage checks]({corpus / 'leakage.json'})",
              f"- [Calibration leakage checks]({corpus / 'calibration-leakage.json'})",
              f"- [Recipe manifest]({corpus / 'recipe/recipe.json'})",
              f"- [Tokenizer manifest]({corpus / 'tokenizer/tokenizer_manifest.json'})",
              f"- [Rebuild verification]({corpus / 'rebuild_verification.json'})",
              f"- [Revised passage-grouping audit]({corpus / 'audited-all-pairs/audit.json'})",
              f"- [Grouping change comparison]({corpus / 'all-pairs-verification.json'})",
              f"- [Revised audit leakage checks]({corpus / 'all-pairs-leakage.json'})",
              f"- [Test verification]({corpus / 'test-verification.json'})",
              f"- [Retrieval diagnostic]({corpus / 'retrieval-diagnostic.json'})",
              f"- [Current environment]({corpus / 'environment-uv.json'})",
              f"- [Cross-version checkpoint check]({corpus / 'environment-portability.json'})",
              f"- [Full evaluation and predictions]({run / 'qualification.json'})",
              f"- [Training result and source exposure]({run / 'training/result.json'})",
              f"- [Saved source manifest]({run / 'training/source/source_manifest.json'})", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
