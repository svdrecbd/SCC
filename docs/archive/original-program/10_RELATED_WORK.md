> Archived document. Its claims and status belong to its original date. See the [current research reset](../../RESEARCH_RESET.md).

# Related Work and Positioning

This file is a starting bibliography, not an exhaustive literature review.

## 1. AntiDote — tamper-resistant LLMs

**Sanyal, Ray & Mandal (AAAI 2026), “AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs.”**

https://ojs.aaai.org/index.php/AAAI/article/view/40570

AntiDote uses bi-level adversarial training in which an auxiliary adversary generates malicious LoRA modifications and the defender learns to resist them. The paper evaluates jailbreak, latent-space, and direct weight-space attacks and reports increased robustness with little utility degradation.

### Relationship to SCC

AntiDote aims for:

`tampering -> safety remains intact`

SCC specifically investigates:

`successful safety removal -> capability collapse`

The methods may overlap, especially bi-level optimization, but the target security property and evaluation geometry differ.

## 2. Safety subspaces are not linearly distinct

**Ponkshe et al. (ICLR 2026), “Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study.”**

https://proceedings.iclr.cc/paper_files/paper/2026/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html

The paper reports substantial overlap between safety-related and useful behavior in both weight and activation space across several open LLMs.

### Relationship to SCC

This is relevant evidence that safety/general-capability representations are not naturally cleanly modular. SCC asks whether training can *increase and weaponize* such overlap into a fail-closed property.

## 3. Deep Ignorance — pretraining-level tamper resistance

**O'Brien et al. (ICLR 2026), “Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs.”**

https://proceedings.iclr.cc/paper_files/paper/2026/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html

The authors pretrain multiple 6.9B models from scratch with selected knowledge filtered from training data and find substantial resistance to adversarial fine-tuning attempting to restore the filtered capability.

### Relationship to SCC

Deep Ignorance is strong evidence that safety-relevant properties can depend on *pretraining-time formation*, not only post-training. Its goal is to prevent acquisition/recovery of a dangerous capability. SCC instead attempts to couple a protected invariant to unrelated indispensable capabilities so that removing the former damages the latter.

## 4. Narrow safety basins during fine-tuning

**Yang et al. (AAAI 2026), “AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin.”**

https://ojs.aaai.org/index.php/AAAI/article/view/40729

This work studies parameter-space directions associated with safety preservation/degradation during fine-tuning.

### Relationship to SCC

SCC can be interpreted partly as an attempt to reshape the safety basin so that trajectories leaving it intersect steep capability loss.

## 5. Watch the Weights

**Zhong & Raghunathan (ICLR 2026), “Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs.”**

https://proceedings.iclr.cc/paper_files/paper/2026/hash/79ee64e62fd3ee763961ce3d6aed4971-Abstract-Conference.html

The work analyzes weight differences between base and fine-tuned models and finds interpretable directions associated with acquired behaviors.

### Relationship to SCC

Useful for designing diagnostics and attackers. If removal behavior is concentrated in identifiable weight-delta directions, SCC should test whether those directions become more capability-critical after coupling training.

## 6. Sleeper Agents

**Hubinger et al. (2024), “Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.”**

https://arxiv.org/abs/2401.05566

The study demonstrates proof-of-concept deceptive/backdoor policies that can persist through supervised, reinforcement, and adversarial safety training.

### Relationship to SCC

Motivates skepticism that behavioral safety training alone necessarily removes strategically embedded policies. SCC does not solve deception directly; it explores a different protection layer against successful model modification.

## 7. Alignment faking

**Greenblatt et al. (2024), “Alignment faking in large language models.”**

https://arxiv.org/abs/2412.14093

The work provides an empirical demonstration of a model varying behavior depending on whether it expects outputs to affect training.

### Relationship to SCC

Motivates separating “the model behaves aligned under training pressure” from “the protected property is structurally difficult to remove while preserving capability.”

## 8. Research positioning statement

A concise positioning for an eventual paper:

> Existing work primarily asks how to preserve safety behavior under model modification or how to prevent dangerous capabilities from being learned. SCC studies a complementary fail-closed property: whether safety-relevant behavior can be deliberately coupled to general capability such that successful safety removal incurs disproportionate capability loss.

## 9. Literature areas to expand

Future review should include:

- machine unlearning;
- model editing;
- catastrophic forgetting;
- parameter-space mode connectivity;
- continual learning;
- neural network watermarking and fragile/semi-fragile watermarks;
- tamper-evident neural networks;
- mechanistic interpretability of safety directions;
- adversarial fine-tuning;
- model merging/interpolation;
- robustness of alignment under continued training;
- feature superposition and representational redundancy;
- information bottlenecks and architectural choke points.
