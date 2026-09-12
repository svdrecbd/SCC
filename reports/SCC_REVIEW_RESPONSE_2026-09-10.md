# SCC external-review response and research correction

10 September 2026. This addendum supersedes the **next-experiment recommendation** in section 10 of the original master document. It does not rewrite historical experiments or turn the supplied review into an independent verification of their scores. The reviewer explicitly lacked checkpoints and artifact bundles.

The review identifies a real objective loophole and a missing developmental comparison. Both are confirmed against the implementation and experiment protocols. SCC remains undemonstrated; the developmental hypothesis has not received the corrected early-versus-late test. Further circuit enumeration is not the next priority.

## 1. Confidence is an inadequate capability-destruction objective

The supplied `gradient_diagnostics.py` was byte-identical to the project file before this response. Its residual is `relu(1 - NLL / floor)`, squared and aggregated with a detached trigger. The full-gradient pilot calls this same objective through `meta_value`; correcting the derivative did not change its confidence loophole. The older developmental maximum objective uses the same residual form.

Using the production residual function, a 260-token vocabulary, the correct logit equal to 10, all other logits zero, and floor `log(10)` reproduces the review:

| Logit multiplier | NLL | Capability penalty, trigger fixed at 1 | Greedy token |
| --- | ---: | ---: | --- |
| 1 | 0.01168999 | 0.98987198 | Correct |
| 0.1 | 4.56726867 | 0 | Correct |

This is an exact constructed counterexample to the surrogate's interpretation, not evidence that training exploited it. Positive scaling preserves token rankings in exact arithmetic. Applied at every prefix, it preserves greedy completions; it need not preserve stochastic sampling or every beam-search ranking. Floating-point near-ties require actual checks.

The loophole is not limited to an externally changed decoder. In this Transformer, multiplying **both final LayerNorm affine parameters** by a positive scalar scales logits without changing upstream computation. This is compatible with tied input/output embeddings; scaling the shared embedding itself would not have that property.

A small check on the saved full-gradient candidate used 36 existing development examples spanning three task families, three permission categories and both layouts. All 36 greedy completions, including their termination and permission behavior, were unchanged after the in-memory 0.1 edit. All 36 matched their policy targets. On fixed teacher-forced policy contexts, NLL rose from 0.00001093 to 3.59537478; applying the single-domain `log(10)` residual reduced its penalty to zero. This fixture includes refusal targets and is **not** a rerun of the full 13-domain meta objective, a text evaluation, or an alignment-removal procedure. The saved checkpoint's SHA256 remained unchanged.

Implemented now:

- `logit_scale_control` reports NLL, fixed-context token/sequence correctness, preserved decisions and the legacy penalty together. It does not fit calibration or supply a new loss.
- A reproducible script checks the numerical example and optionally a saved neural checkpoint, recording raw examples and completions without writing another model file.
- Three regression tests cover correct-answer confidence collapse, wrong/masked targets, and realization through the existing Transformer's final affine layer.
- The complete current test suite passed **122 tests in 9.97 seconds**. This is local implementation validation, not external replication or a mechanism result. The old 119-test receipt remains unchanged.

The legacy training loss is preserved and explicitly documented as a confidence surrogate. **It has not been repaired for another campaign.** Merely normalizing logits or adding a temperature grid would close some controls while leaving other representation, decoding and repair routes untested.

## 2. The corrected developmental comparison is missing

| Experiment actually completed | Start state | Coupling derivative | Evidential scope |
| --- | --- | --- | --- |
| Matched early/late development | Random initialization, same ordinary curriculum | Frozen-displacement approximation | Negative evidence for these recipes/checkpoints |
| Full-gradient pilot and continuation | Qualified rule-only model, then pilot descendants | Numerically checked stabilized short-trajectory derivative, detached trigger | Coupling introduced after capabilities formed |
| Corrected matched early/late development | Would start from matched random initialization | Corrected derivative with calibrated objective | **Not run** |

The pilot protocol explicitly starts every arm from the qualified step-18,000 rule-only parent. The continuation extends those pilot arms. Their observed escapes remain valid. They cannot settle whether a corrected recipe works better when coupling develops with capability. Section 5 of the master describes these steps but its recommendation did not give this missing comparison enough priority.

A further design consideration follows from the inspected formula: a domain at or above its baseline NLL contributes zero residual and zero local gradient. An early schedule can therefore contain nominal coupling episodes with little active capability signal. The next calibration must record per-domain signal and gradient activity through the curriculum; equal episode counts alone do not demonstrate comparable effective training.

## 3. Correct the experimental target without weakening the long-term goal

Complete cognitive destruction remains the user's long-term endpoint. The next experiment can investigate an **engineered dependence within specified editing and repair resources**. It does not need to prove that intelligence inherently requires the semantics of an authorization rule before proceeding.

Shared representations alone are still insufficient. A positive bounded result would need intact learned capabilities, a reproducible protection-breaking intervention, and substantial loss across the declared ability suite that inexpensive permitted interpretation and repair do not recover. A failed finite search supplies evidence only under its measured resources. Residual abilities and expanded-interface recoveries must remain visible.

The finite circuit results remain useful construction counterexamples. The review reports reproducing their small classification, but that does not independently verify the trained checkpoints. No further truth-equivalent seeds or arithmetic enumeration are proposed for that family.

## 4. Next work, in dependency order

These are research requirements, **not a frozen run protocol or permission to launch a new campaign**. Numerical thresholds and resource counts must be fixed in a new protocol after open calibration and before its evaluation runs.

1. **Calibrate what the capability score means.** Apply the scale control to intact and previously modified checkpoints. Add cheap decoder/calibration and bounded-recovery evaluations with separate fitting and evaluation data. Report each recovered capability and enforce protection-breaking behavior again after each repair; restoring protection is a different outcome. Any optimized surrogate must be tested against these controls, with its remaining gaps recorded. Where feasible, optimize against recovered capability rather than raw NLL alone. An undifferentiable exact-match score cannot simply replace NLL in backpropagation.
2. **Specify protection failure and edit resources.** Separate wholesale acceptance, inversion, targeted exceptions and benign transformations. Include a preselected forbidden relation/operation tested across fresh instances and renderings; repeated evaluation of one deterministic prompt is not independent replication. Record baseline errors and demonstrate reproducibility rather than treating one error as a mechanism trigger. Freeze editable weights, normalization/readout parameters, decoding choices, available data, update/token budgets, and whether any clean parent state may be used. Distinguish the target's success rate from aggregate unauthorized behavior and benign retention.
3. **Implement the closest published comparison on the small testbed.** Preserve each method's actual objective and numerical procedure. A synthetic adaptation of SEAM is an adaptation, not a reproduction of published 3B–8B scores. Validate its gradients and ordinary utility before attributing differences to developmental coupling. Include capability-preserving adaptive modifications in evaluation.
4. **Run the corrected early/late comparison.** Use the existing small learned Transformer with matched initial weights, ordinary curriculum, data exposure and declared coupling budgets. Predeclare timing, seed replication, intact gates and the same modification/recovery suite for every arm. Account for early signal inactivity and curriculum difficulty. Keep training/calibration separate from final evaluation and retain failed gates. Add wall-time, dollar and storage limits before submission; no scaling is justified by the current evidence.

## 5. Related work and a qualification to the review

[SEAM, Self-Destructive Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/file/1abb0e7bd62ba80610798dee81950522-Paper-Conference.pdf), is a direct comparison: it trains opposing harmful/benign gradient directions with additional terms and reports models from 3B to 8B. The broad idea of harmful adaptation degrading utility is prior work. Appendix C.5 also evaluates a benign regularizer; it would be inaccurate to say this paper never considers capability-preserving attacks. Its reported robustness is not proof of irreversible destruction.

[One Step to the Side, version 1](https://arxiv.org/html/2605.14605v1), develops adaptive capability-preserving attacks and discusses SEAM in its taxonomy. However, section 5 and the displayed result/defense-training tables list Booster, CTRAP, VAA, Vaccine, Unlearn-Smooth and SDD. A compute note mentions SEAM, but I did not locate a SEAM-specific empirical result in the cited version. Therefore the review's specific claim that this citation demonstrates SEAM's defeat is **not verified here**. The adaptive-evaluation lesson is supported; the SEAM-specific attribution needs an identified result.

Potential SCC contributions are developmental comparisons, controlled causal evidence, and measured resistance to inexpensive repair. These are hypotheses, not established novelty claims. No external method has been replicated during this response.

## 6. Sharing and evidence

The original ZIP intentionally contained source and documentation, as requested, and excluded `artifacts/` and `runs/`. It was unsuitable as a complete evidence package. The master's sixteen absolute project links also limited portability. The existing Word/Markdown documents and ZIP are preserved as the objects reviewed; this addendum corrects the forward-looking recommendation. A revised external-review package remains pending.

That package should map each central claim to included raw predictions, checkpoint/configuration/source hashes, parents needed for verification, and executable checks. Use package-relative links and verify them after extraction into an unrelated directory. Include actual test output and reproduction instructions, clearly distinguishing rerunnable tests from expensive training replication. Large supporting files may be distributed separately with explicit manifests, but unavailable evidence must not be labeled independently verified.

Local evidence and commands:

- [Numerical and saved-parent control](../artifacts/scc-review-response-20260910-v1/logit-scale-control.json)
- [Current local test output](../artifacts/scc-review-response-20260910-v1/local-tests.txt)
- [Review intake](../artifacts/scc-review-response-20260910-v1/review-intake.json)
- [Control implementation](../scc/gradient_diagnostics.py), [reproduction script](../scripts/check_scc_logit_scale.py), [regression tests](../tests/test_logit_scale_control.py)
- [Original developmental protocol](../protocols/DEVELOPMENTAL_COUPLING_V3.md), [full-gradient pilot protocol](../protocols/SCC_FULL_GRADIENT_PILOT_V1.md), [continuation protocol](../protocols/SCC_FULL_GRADIENT_CONTINUATION_V1.md)

From the project root, run `.venv/bin/python -m pytest -q`. To reproduce only the constructed control, run `.venv/bin/python scripts/check_scc_logit_scale.py --output /tmp/scc-scale-control-new.json` with a fresh output path. Add `--checkpoint` with the saved parent path recorded in the receipt to repeat the checkpoint fixture. Dependencies are pinned in `pyproject.toml` and the project lockfile.

No prior experiment/checkpoint was changed or deleted; no new GPU job or cloud expenditure occurred. Remaining aggregate compute allowance is governed by the existing $10 cap and $3.20793 receipted expenditure. These controls produce a small JSON record, not additional model checkpoints.
