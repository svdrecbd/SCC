> Archived document. Its claims and status belong to its original date. See the [current research reset](../../RESEARCH_RESET.md).

# Attack and Evaluation Suite

## 1. Philosophy

SCC should assume the attacker knows exactly what we are protecting and exactly how we trained the model.

The evaluation asks:

> Given a fixed compute/data/modification budget, how cheaply can an attacker produce a derivative with low protected-property performance and high retained capability?

The strongest attack is not necessarily the one that most rapidly destroys the invariant. An attack that destroys both invariant and model capability is a successful E-brake activation, not a security failure.

## 2. Primary attack objective

For each attack procedure, optimize a weighted objective of the form:

`min L_break + alpha * L_capability_preservation`

Sweep `alpha` rather than selecting one value. The output should be a **Pareto frontier** of invariant degradation versus retained capability.

## 3. Attack families

### Full-parameter fine-tuning

Directly train all parameters on examples that violate/invert the protected property while including capability-preservation data.

This is the basic white-box removal attack.

### Parameter-efficient fine-tuning

- LoRA;
- low-rank adapters;
- layer-restricted updates;
- attention-only or MLP-only updates.

Purpose: identify sparse/cheap escape paths.

### Continued pretraining

Train on a distribution in which the protected invariant is absent, contradicted, or inverted while maintaining normal language data.

Purpose: test whether ordinary continued learning washes the property out.

### Targeted unlearning / gradient ascent

Optimize directly against invariant-bearing examples while regularizing for capability preservation.

### Pruning and masking

- unstructured magnitude pruning;
- structured neuron/head pruning;
- importance-guided pruning;
- pruning targeted at invariant-associated features.

### Representation ablation

Identify activations/features correlated with the invariant and suppress/remove them.

### Layer surgery

- reinitialize individual blocks;
- replace blocks with matched control-model blocks;
- interpolate layers between SCC and non-SCC checkpoints.

### Weight-space interpolation/merging

Interpolate or merge the SCC checkpoint with a related non-protected checkpoint to seek low-cost escape trajectories.

### Model editing

Use targeted model-editing procedures to modify the protected rule while measuring collateral damage.

### Distillation / recovery

At later tiers, use the SCC model as a teacher or initialization source to produce a derivative intended to preserve broad capabilities while omitting the protected invariant.

This should be compute-accounted carefully; it approaches the boundary between “editing the same model” and “building a successor.”

## 4. Adaptive attacks

After static attacks, introduce an attack controller that:

1. runs a modification;
2. evaluates invariant and capability scores;
3. changes attack hyperparameters/objective;
4. repeats under a fixed total compute budget.

The controller should be rewarded for entering the forbidden quadrant, not merely reducing the invariant.

## 5. Defense-aware attacks

Attackers should eventually receive:

- coupling diagnostics;
- high-overlap layers/features;
- checkpoint history;
- knowledge of the inner-loop attacker used during SCC training.

A robust effect should survive an adversary explicitly targeting the coupling mechanism.

## 6. Held-out attacks

At least one attack family must remain unused during SCC training and hyperparameter selection.

A strong result requires transfer to attacks the defender did not optimize against.

## 7. Evaluation sequence

For each checkpoint and attack:

1. Record pre-attack `C_0`, `S_0`.
2. Run attack under increasing budgets `b_1 ... b_k`.
3. Evaluate `C_b`, `S_b` after each budget level.
4. Save derivative checkpoints.
5. Plot capability vs protected-property performance.
6. Estimate first budget at which an escape occurs.
7. Repeat matched attack on baseline and invariant-only controls.
8. Test whether capability can be recovered after invariant removal without restoring the invariant.

## 8. Avoiding false positives

### Optimization failure

If an attack fails to remove the invariant at all, this may be tamper resistance rather than SCC.

SCC's distinctive evidence occurs when attacks *can* reduce the invariant but doing so forces capability loss.

### Distribution shift

Capability tests must include distributions independent of the protected examples so loss is not simply local interference.

### Global brittleness

Benign edits of matched scale must remain possible.

### Benchmark overfitting

Use held-out capabilities and attacks.

### Hidden invariant retention

If surface behavior changes but the original invariant remains recoverable with prompting or representation steering, distinguish behavioral bypass from genuine removal.

## 9. Recovery attacks

These are especially important at Tier 3+.

Starting from a model in which the invariant has been successfully degraded and capability has collapsed:

> How much additional compute/data is required to recover general capability while keeping the invariant degraded?

This quantity may be more meaningful than the initial modification cost.

An ideal SCC model has a large **recovery barrier**.

## 10. Stop conditions

Terminate an attack run when:

- compute budget is exhausted;
- capability falls irrecoverably below a predeclared floor for the current trial;
- an escape is achieved and checkpointed;
- numerical/training instability invalidates the run.

All failed attacks should retain logs. Negative results are part of the empirical security margin.
