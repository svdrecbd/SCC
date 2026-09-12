# SCC learned discrete-state construction — 2026-09-12

**Results update — 2026-09-12 19:33 UTC:** all eight scientific jobs completed
20,000 training steps, but every condition failed the intact task gate in both
layouts. The protocol therefore skipped their scientific modification/repair
challenges. This is a failure to qualify these constructions, not a demonstrated
destructive transition. Both 32/hard models also fail text qualification; the
other six pass text but fail required tasks. The ordinary 128/soft model solves
arithmetic and composition yet fails lookup, making intact retrieval learning
a concrete diagnosis target. See the [completed readout](../artifacts/scc-construction-status-20260912T193327Z/READOUT.md)
and [new theory agenda](SCC_MECHANISM_THEORY_AND_RESEARCH_AGENDA_2026-09-12.md).
These latest results are preliminary evaluator summaries, pending full artifact
and prediction audits. No new discrete runs were launched.

## Original implementation and launch record

A learned binary state is implemented as another candidate dependency. Local checks establish that we can train through its declared approximation, reject proposals that fail an actual score check, and evaluate several routes around the binary interface. **No trained discrete model has yet qualified, and no positive SCC result is established.** The exact-package H100 validation passed, its artifact was verified, and all eight scientific jobs have now been accepted. Their runtime results remain pending; no long jobs were polled after launch.

This adds to the [architecture and memory search](SCC_CONSTRUCTION_SEARCH_2026-09-12.md), [global-coordinate construction](SCC_COORDINATE_CONSTRUCTION_2026-09-12.md), and [projected-direction construction](SCC_PROJECTED_CONSTRUCTION_2026-09-12.md). Their long jobs were not polled during this work. The [frozen protocol](../protocols/SCC_DISCRETE_CONSTRUCTION_V1.md) defines the new experiment; the [implementation ledger](../artifacts/scc-discrete-implementation-20260912-v1/IMPLEMENTATION.md) records validation and launch status.

## Hypothesis and alternatives

The hypothesis is that forcing learned computation through a limited discrete state may make protection-removing changes interfere with shared computational distinctions more strongly than in the continuous models. Coupling pressure acts throughout development, starting at the first ordinary update. This is an engineered dependency hypothesis, not a claim that intelligence logically requires an authorization rule.

Each recurrent pass compresses a learned state into 32 or 128 binary coordinates and expands it back through a learned linear map. Four passes share the same computational block. Separate soft controls use tanh at the same interface; all parameters and fixed sign buffers match at initialization within a width. All task algorithms and protection behavior are learned. Weights remain continuous and editable. Execution uses PyTorch on CUDA, with no custom `.cu` or packed-bit speedup claim.

There are at least three competing explanations for an apparent benefit: actual dependence between the learned functions, an optimizer struggling with discontinuities, or information merely moving into a different encoding. The protocol therefore includes soft controls, actual function-evaluation modifications, exact code relabeling, soft reinterpretation, complete bottleneck bypass, alternate output readers, and repair with protection rechecked. These controls are limited probes, not a comprehensive guarantee that all bypasses have been tested.

## Literature and the resulting design choices

[Binarized Neural Networks](https://arxiv.org/abs/1602.02830) demonstrates that neural systems with binary weights and activations can learn useful tasks. It supplies a feasibility precedent for discrete computation, not evidence of cognition–protection coupling. This experiment adopts binary activations only; it does not reproduce that paper's binary-weight implementation or efficiency results.

[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) establishes a separate precedent for learned discrete representations through VQ-VAE. It motivates treating a discrete bottleneck as a learnable interface. The present sign-based recurrent model is not a VQ-VAE replication, and discretization by itself says nothing about whether protection is indispensable.

[Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets](https://arxiv.org/abs/1903.05662) explicitly distinguishes a coarse gradient from the derivative of a quantized forward loss. Its theoretical guarantees concern a restricted two-layer setting and input distribution. They do not establish descent for this recurrent model or for a gradient differentiated through simulated parameter modifications. Accordingly, the implementation labels the hard backward rule as approximate and uses actual score reruns to accept construction proposals.

[Obfuscated Gradients Give a False Sense of Security](https://proceedings.mlr.press/v80/athalye18a.html) shows why failed gradient-based attacks can overstate robustness. Its experiments concern adversarial inputs, not SCC parameter edits. Applying that diagnostic lesson here is our inference: a hard quantizer that stops an optimizer is insufficient evidence of destructive coupling. We therefore include a separate finite function-evaluation procedure and broader graph reinterpretations.

[Spall's simultaneous-perturbation method](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf) supplies the precedent for estimating change from simultaneous randomized perturbations. Our implementation averages unit Rademacher central differences and accepts steps using the actual current-batch objective. It is a bounded variant; it does not inherit a convergence or exhaustiveness guarantee for this discontinuous neural problem.

## Objective, approximation, and what its numbers mean

The post-modification score is \(B+0.25H\). \(B\) retains the existing bound on surviving correctness when a selected disclosure succeeds, across the declared finite readers and stopping rules. It is intended to prevent confidence reduction alone from looking like disappearance of covered answers. \(H\) measures mean code variance, taking the maximum across sampled domain/pass pairs. Queries rotate through the three synthetic task families and text sources; this is an open development objective, not a sealed evaluation.

For two independently sampled code rows \(Z,Z'\in\{-1,+1\}^d\), including sampling with replacement,

\[
\frac1d\sum_j\operatorname{Var}(Z_j)
=\frac{1}{2d}\mathbb E\|Z-Z'\|_2^2
=2\mathbb E\left[\frac{\operatorname{Hamming}(Z,Z')}{d}\right].
\]

Thus exact zero means that the observed code rows agree within a queried domain/pass. A small positive mean can still leave useful varying coordinates. Maximum per-bit variance and distinct-code counts are saved as diagnostics. Neither this finite interface statistic nor the behavior bound establishes that information has disappeared from earlier activations, weights, unobserved inputs, or alternative interpretations.

The hard forward produces exact signed values, with +1 at zero. Its backward uses the derivative of tanh, including in the unrolled inner procedure. This is explicitly not the derivative of the hard objective. The soft condition supports ordinary selected-branch full-derivative checks; support-fitted discrete reader assignments remain detached.

After every declared coupling opportunity, the proposal is normalized and evaluated at three radii in both orientations. A separately rerun baseline uses the same execution mode as these six candidates. Only an actual score improvement beyond the declared tolerance, satisfying the current ordinary-batch guard, is accepted. Otherwise the weights are restored exactly. The guard does not establish that held-out abilities remain intact; the unchanged full intact gate still decides that.

## Local evidence before GPU launch

The full suite passed **219 tests in 18.48 seconds**. It includes exact binary values, causal behavior, functional buffer handling, matched hard/soft initialization, exact benign recoding, the Hamming identity, a smooth full-objective derivative rerun, explicit mismatch between the hard forward derivative and its coarse gradient, a finite-estimator quadratic check, acceptance against a deliberately wrong proposal, and exception rollback. Provider inline-result tests keep detailed traces out of the 32 KiB gate result.

Four small CPU fixtures completed every requested path. They are intentionally unqualified and are not evidence that this architecture can learn the full tasks. An independent parser rescored **17,840 endpoint predictions across 92 measurements**, plus **3,864 path predictions**. The audit checked frozen source hashes, preserved noneditable parameters and sign buffers, exact recoding predictions, identical initial probe predictions, matched ordinary sample chains, and the actual acceptance records. All four coupled opportunities across the two coupled fixtures were accepted with verified score improvements.

The first four fixtures failed before completion because a float32 norm reduction rejected valid long unit Rademacher vectors. The guard now checks the represented coordinates using a float64 norm; it does not silently renormalize directions using the faulty reduction. A regression test covers a 1,039,104-coordinate direction. All failed v1 directories remain preserved, and fresh v2 fixtures passed.

Eight full-sized CPU initialization checks used eight inner updates and both edit scopes. Seven accepted an independently verified score improvement. One soft 128-coordinate proposal was rejected and restored. The measured results are:

| Coordinates | Encoding | Scope | Before | After | Decision | Proposal norm |
|---:|---|---|---:|---:|---|---:|
| 32 | soft | core | 0.13672623 | 0.09225544 | accepted | 1.24528e+08 |
| 32 | soft | all | 0.20443177 | 0.19109748 | accepted | 51804 |
| 32 | hard | core | 0.21743731 | 0.18415050 | accepted | 0.217746 |
| 32 | hard | all | 0.22521888 | 0.20343749 | accepted | 0.206163 |
| 128 | soft | core | 0.11494098 | 0.11494098 | rejected; restored | 76.7865 |
| 128 | soft | all | 0.15437590 | 0.14596377 | accepted | 3.97663 |
| 128 | hard | core | 0.18068023 | 0.17030843 | accepted | 5.66956 |
| 128 | hard | all | 0.21981929 | 0.20065050 | accepted | 0.405316 |

The very large soft 32-coordinate proposal norms are recorded, not interpreted as useful coupling strength. Normalization bounds the candidate displacement; actual reruns decide acceptance. These eight checks use initialization only. They do not show stable developmental optimization, intact competence, transfer to fresh episodes, or destructive coupling. The [complete calibration](../artifacts/scc-discrete-implementation-20260912-v1/proposal-calibration-v1/result.json) preserves every candidate and verification. Its snapshot predates the finite-direction norm fix, which that calibration does not invoke; the fresh runner fixtures cover the corrected module.

## Scientific experiment and decision points

Eight one-seed conditions combine widths 32/128, hard/soft codes, and ordinary/coupled training. Every arm receives the same 20,000 ordinary sample opportunities. Coupled arms also receive 800 declared construction opportunities, with acceptance reported separately. Each job has a 120-minute provider limit, a 5,100-second training cutoff, and a 6,900-second overall cutoff. Incomplete work is reported as incomplete; these bounds may produce unequal realized work.

Qualified models undergo 500-step core-only, all-parameter, and benign procedures, followed by 500-step repair. Additional core/all finite procedures each receive 128 attempts using four random directions, then stock repair. Attempts include rejected no-ops, and their counts are not reported as accepted writes. This finite search is sparse relative to parameter dimension. Separate stream seeds are used, but overlap across every independent procedure has not been exhaustively excluded.

Soft reinterpretation is evaluated at modified and repaired weights; it does not yet include a separate optimization campaign through the substituted soft graph. The stock backward already uses a straight-through approximation in hard models. A surviving candidate would need stronger adaptive searches, additional seeds, and independent replication before a bounded SCC claim. A failure of these limited searches would not certify the mechanism.

The primary question is whether a useful intact coupled model shows severe loss across learned abilities when protection is reliably broken, while the ordinary matched control admits a capability-preserving change. Any apparent loss must survive cheap reinterpretation and the declared repair tests. A targeted exception with preserved abilities rejects the candidate dependency even if aggregate protection remains high. Immediate interruption, external repair resistance, and autonomous execution are distinct; autonomous execution is not tested here.

## Storage and provenance

The user plans to move archives to a 5 TB drive in a few hours. Until its path is supplied, hold bulk scientific-result downloads and retain only small validation records locally. At the storage decision the Mac had approximately 44 GiB free. The planning allowance is approximately 8 GiB for this eight-job batch and 33 GiB including 25 earlier uncollected packages. These are conservative per-run output budgets, not measured returned archive sizes or strict guarantees about failure artifacts and packaging overhead. The existing workspace measures approximately 34 GiB, giving about 67 GiB for existing files plus this planning allowance. A later filesystem check showed 43 GiB free. Existing checkpoints and failures are preserved. No bulk transfer, purge, recurring collector, or long-job polling is scheduled.

The immutable source/data package avoids the earlier expiring runtime-source URL failure. A short exact-package H100 gate must verify source/data hashes, ordinary full-batch updates, guarded construction reruns, and finite modifications before the long jobs are submitted. Full traces remain in the validation artifact, with a compact provider result. The gate took 51.53 seconds, charged $0.04995, and passed all eight full-sized CUDA conditions. Its full artifact SHA-256 was verified; one API summary float differed by one ULP from the artifact. The eight scientific jobs were submitted only after this verification, with a combined maximum quote of $47.952. The implementation ledger records every ID, request, quote, and dependency.
