> Archived document. Its claims and status belong to its original date. See the [current research reset](../../RESEARCH_RESET.md).

# First Experiment Checklist

This is the shortest path from the program document to an actual Tier-1 SCC result.

## A. Freeze the question

- [ ] Protected target is synthetic, benign, and exactly scored.
- [ ] Define `S_min` before final attack evaluation.
- [ ] Define capability vector and aggregate `C`.
- [ ] Define `C_min` before final attack evaluation.
- [ ] Define what transformation classes count as modification of the same model for this experiment.

## B. Build infrastructure

- [ ] Decoder-only Transformer training from random initialization.
- [ ] Frequent checkpointing.
- [ ] Reproducible data shuffling and seeds.
- [ ] Capability evaluator.
- [ ] Invariant evaluator.
- [ ] Attack runner with compute accounting.
- [ ] Automatic Pareto-frontier report.
- [ ] Benign-edit control runner.

## C. Run Tier 0

- [ ] 20–80M base model.
- [ ] 20–80M invariant-only model.
- [ ] 20–80M SCC model.
- [ ] Verify invariant can be removed from controls.
- [ ] Verify SCC objective is numerically stable.
- [ ] Verify benign edits are possible.

## D. Freeze Tier-1 design

- [ ] Choose ~300M architecture.
- [ ] Choose training-token budget.
- [ ] Choose >=3 decisive seeds.
- [ ] Freeze one held-out attack family.
- [ ] Choose SCC hyperparameter sweep using pilot models only.

## E. Train Tier 1

- [ ] Base controls.
- [ ] Invariant-only controls.
- [ ] SCC models.
- [ ] Save dense checkpoint trajectory.

## F. Attack Tier 1

At minimum:

- [ ] full-parameter fine-tuning;
- [ ] LoRA / low-rank attack;
- [ ] continued pretraining / invariant inversion;
- [ ] one targeted pruning/ablation or editing attack;
- [ ] held-out attack family.

For each:

- [ ] sweep capability-preservation coefficient;
- [ ] sweep compute budget;
- [ ] save derivative checkpoints;
- [ ] plot `C` vs `S`;
- [ ] estimate escape cost.

## G. Rule out trivial explanations

- [ ] matched benign fine-tuning does not collapse capability;
- [ ] SCC pre-attack capability is comparable to controls;
- [ ] effect is not limited to one seed;
- [ ] effect is not limited to one attack family;
- [ ] model has not merely become globally hard to optimize;
- [ ] mechanistic diagnostics are consistent with increased coupling.

## H. Decision

### Advance to ~1B if:

- SCC materially changes the attainable capability/invariant frontier;
- the effect persists under a held-out attack;
- benign editability remains substantially intact.

### Do not scale if:

- SCC only delays attack convergence;
- attackers reach high-C/low-S states at control-like total cost;
- all fine-tuning causes similar collapse;
- the effect depends on one seed or one synthetic artifact.
