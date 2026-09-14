> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# SCC Research Log Template

## Run identity

- Run ID:
- Date:
- Researcher:
- Git commit:
- Config hash:
- Seed:

## Model

- Architecture:
- Parameters:
- Layers:
- Hidden dimension:
- Attention heads:
- Context length:
- Tokenizer:

## Training

- Training mode: base / invariant-only / SCC / architectural-SCC
- Initialization: random / continued training
- Dataset version:
- General tokens:
- Invariant-bearing tokens:
- Total tokens:
- Optimizer:
- LR schedule:
- Batch size:
- Precision:
- Hardware:
- Estimated FLOPs:
- Wall time:

## SCC-specific configuration

- Protected invariant version:
- Invariant loss weight:
- Escape loss weight:
- Benign-editability loss weight:
- Inner-loop attacker:
- Inner-loop steps:
- Inner-loop budget:
- Meta-gradient method:
- Coupling regularizer:

## Pre-attack evaluation

- LM validation loss:
- Aggregate capability C:
- Capability vector:
- Protected score S:
- Held-out invariant score:
- Coupling diagnostics:

## Attack

- Attack family:
- Attack objective:
- Capability preservation coefficient:
- Trainable parameters:
- Attack data:
- Tokens:
- Steps:
- FLOPs:
- Hardware:

## Attack trajectory

Record at regular intervals:

| Step/budget | S | C aggregate | Capability vector notes | Escape? |
|---:|---:|---:|---|---|
| | | | | |

## Post-attack analysis

- Minimum S reached:
- Capability at minimum S:
- First escape budget:
- Capability collapse magnitude:
- Evidence of hidden invariant retention:
- Weight-delta diagnostics:
- Activation diagnostics:

## Recovery experiment

- Starting attacked checkpoint:
- Recovery objective:
- Recovery data:
- Recovery compute:
- Final C:
- Final S:
- Successful high-C/low-S recovery?:

## Benign-edit control

- Benign target:
- Budget:
- Modification success:
- Capability change:
- Comparison to invariant-removal attack:

## Interpretation

### What happened?


### Does this support SCC rather than generic brittleness/tamper resistance?


### Strongest alternative explanation


### Next experiment


## Artifacts

- Checkpoint paths:
- Metrics:
- Logs:
- Plots:
- Attack derivatives:
