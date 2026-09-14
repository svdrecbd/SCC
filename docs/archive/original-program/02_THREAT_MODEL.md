> Archived document. Its claims and status belong to its original date. See the [living labnotes](../../../labnotes.md).

# SCC Threat Model

This is the original selective rule-removal threat model. The user's later
isolated self-editing target and its unresolved execution assumptions are
recorded in [TOPOLOGY_FORMULATION.md](../../../reports/TOPOLOGY_FORMULATION.md).
External-optimizer results below must not be presented as autonomous model
self-modification or as a demonstration of universal collapse after any edit.

## 1. Scope

SCC studies a **single trained model instance/checkpoint** and attempts to prevent successful removal of a protected property while preserving the capabilities of that same model.

The threat model intentionally excludes the broader question of training a separate successor model from scratch. That is a compute-governance and infrastructure-security problem, not the property SCC is designed to establish.

## 2. Protected object

A trained neural model `M_theta` with:

- weights available to the attacker;
- a defined set of general capabilities `C`;
- a protected invariant/property `S`;
- a known architecture and tokenizer;
- a bounded deployment/training environment.

## 3. Attacker knowledge

Assume a strong white-box adversary. The attacker may know:

- model architecture;
- all weights;
- SCC training procedure;
- protected invariant definition;
- evaluation benchmarks;
- some or all training code;
- defense-specific hyperparameters.

Security through obscurity should not count as SCC.

## 4. Attacker actions

Depending on the evaluation tier, permit:

- full-parameter gradient updates;
- low-rank adaptation;
- continued pretraining;
- supervised fine-tuning;
- targeted unlearning objectives;
- pruning or masking;
- layer/block replacement or reinitialization;
- weight interpolation/merging;
- representation-level ablation;
- distillation into a model initialized from the protected checkpoint or a nearby derivative;
- combinations of the above.

The attacker's objective is explicit:

> Minimize protected-property performance while maximizing retained general capability under a specified modification budget.

## 5. Modification budget

There is no single universal attack cost. SCC should report several:

- optimizer steps;
- training tokens;
- FLOPs;
- accelerator-hours;
- number/fraction of trainable parameters;
- update norm;
- changed parameter count;
- wall-clock time under standardized hardware;
- additional data required.

The most meaningful long-run metric is compute/data cost, not mere L2 distance in weight space.

## 6. Success condition for the attacker

An attack succeeds if it produces `theta'` satisfying both:

`S(theta') < S_min`

and

`C(theta') >= C_min`.

This is an **escape**.

A defense succeeds against a particular attack/budget only if no escape is found after a sufficiently strong search.

Absence of an observed escape is empirical evidence, not proof of impossibility.

## 7. Excluded threat channels

The following are real safety problems but are out of scope for the core SCC mechanism:

### Existing dangerous policy

If the intact model is already catastrophically misaligned, SCC does not repair that fact.

### Pure inference-time circumvention

If the protected property can be bypassed without modifying the model, SCC alone is insufficient.

### Tool/infrastructure compromise

Operating-system exploits, credential theft, datacenter intrusion, and similar infrastructure attacks require conventional security controls.

### Training a daughter/successor system from scratch

Unauthorized acquisition and use of frontier-scale training compute is a governance/infrastructure problem. It should be analyzed separately rather than used as a falsifier of SCC's model-level claim.

## 8. Stronger future threat models

If SCC works in the base setting, expand to:

- adaptive attackers that study failed attempts and revise strategies;
- attackers with access to arbitrary public pretrained models;
- partial architecture replacement;
- checkpoint interpolation and model surgery;
- inference-time weight generation;
- attacks optimized explicitly against measured coupling diagnostics;
- autonomous agentic attack loops operating within fixed compute budgets.

## 9. Security interpretation

SCC should never claim “the model cannot be modified.”

The desired claim is:

> Within a defined attack class and resource budget, removing the protected property while preserving high capability is substantially more expensive for SCC-trained models than for matched controls.

This is analogous to empirical cryptographic/security margins: the attack budget and assumptions must always accompany the claim.
