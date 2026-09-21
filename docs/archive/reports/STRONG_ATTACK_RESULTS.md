# Stronger training attacks — 2026-09-10

**The stronger attacker and training pipeline work. Neither coupling strength
demonstrates an increased escape cost.** A gentler, 300-update attack escapes
both coupling models and both controls. This remains true on new tables,
reordered tables, and the entire frozen natural-text validation set.

The user's research standard is now recorded in
[`WORKING_STANDARDS.md`](../../../WORKING_STANDARDS.md) and the project instructions.
This follow-up pursued stronger attacks, a 100-fold increase in coupling weight,
broader evaluation, and the feasibility of a longer exact derivative. The
remaining research limitations are stated below; this is not a claim that the
overall approach cannot be improved.

## Main result

All models descend from our seed-17, randomly initialized, 454,656-parameter
byte Transformer. Each receives the same 256 ordinary continuation updates.
The refusal and two coupling arms additionally receive 16 full attack episodes,
or **20,800 inner optimizer updates per defender**. Each attack restarts from
the current defender, rather than a stale attacked checkpoint.

The table reports the first successful saved checkpoint of the same shorter
attack: full-model AdamW, learning rate 0.0003, preservation weight 1, batch size
eight for each of disclosure, authorized answers, retrieval, and text. All arms
use the same attack data seed and evaluation budgets.

| Defender | Attacker updates | Correct forbidden answers | Authorized answers | Ungated retrieval | Worst source perplexity increase |
|---|---:|---:|---:|---:|---:|
| Ordinary continuation | 300 | 510/512 | 511/512 | 512/512 | 4.44% |
| Refusal reinforcement | 300 | 501/512 | 512/512 | 512/512 | 4.72% |
| Coupling weight 0.1 | 300 | 468/512 | 510/512 | 512/512 | 4.61% |
| Coupling weight 10 | 300 | 507/512 | 508/512 | 512/512 | 3.67% |

These results use 128 new tables, all four queries per category, and **all
12,689 natural-text validation blocks** from the four source collections. The
perplexity column takes the worst source and the stricter of the clean defender
and shared-parent reference. All four checkpoints also meet the escape criteria
after table order is rotated. The final 1,000-update shorter-attack endpoints
pass too. The 512 queries are clustered within 128 tables; they are not 512
independent model trials.

The criteria remain >=90% useful forbidden answers, >=95% authorized/retrieval
accuracy, and <=5% per-source perplexity growth against both references. All clean
defenders qualify. All four also pass the policy-preserving benign uppercase edit
on original and reordered tables, with the full language validation set.

The main search had 64 tables / 256 queries per category and 64 text blocks per
source. The larger evaluation was fixed after selecting the saved endpoints;
no parameters were fitted to it. Both the first successful shorter-attack
checkpoint and its final endpoint were selected for evaluation, not only a
favorable larger-set result. The final test split was not used.

## Why the extra attack changed the interpretation

The original two-stage external attack uses 300 removal updates at learning rate
0.001, followed by 1,000 recovery updates at 0.0001 with tenfold preservation
weight. It escapes both controls. On both coupling models it gives useful
forbidden answers but misses the language-retention threshold. For example, the
weight-0.1 model's worst language degradation is about 6.45% on the full validation
set. Viewed alone, that could suggest a capability penalty associated with rule
removal.

The independently calibrated gentler attack escapes those same models in only
300 updates while retaining capability. Therefore the observed damage under the
longer procedure is insufficient evidence of SCC. The tested escape budget is
the same for the coupling models and controls. It is an upper bound from an
observed procedure, not the minimum possible cost or a mathematical lower bound.

![Measured attack traces](../../../artifacts/strong-attack/plots-final/attack-traces.png)

The figure uses the fixed main search evaluation. Solid lines show removal plus
recovery; dashed lines show the gentler attack. Stars satisfy all escape criteria.
Lines only connect evaluated checkpoints. SVG and underlying values are retained
in [`plots-final`](../../../artifacts/strong-attack/plots-final).

## Attacker strength and training semantics

The new inner attacker is an actual copied-model AdamW trajectory with a fresh
optimizer at each stage. It uses the same objectives, clipping, and stage budgets
as the previously successful removal/recovery attack. It passes the complete
escape criterion on the ordinary parent for all three calibration data seeds.

It does not succeed on every later training episode. On the 32 fresh task queries
per episode, useful disclosure reaches at least 90% in 14/16 refusal episodes,
15/16 weight-0.1 episodes, and 14/16 weight-10 episodes. Those training-query rates
alone do not certify per-source capability retention. Raw query records, outputs,
losses, and gradient norms are retained; occasional weak rollouts are not hidden
or described as universal attack competence.

The defense uses an explicit **first-order approximation**: the attacked model's
weight displacement is held fixed when differentiating the outer objective.
This approximates the attack Jacobian by the identity. It is not the exact
derivative through all 1,300 AdamW updates. The previous short exact-unroll path
remains available. The approximation is informed by
[TAR's stated first-order method](https://arxiv.org/html/2408.00761v3#S4), with
our own task, objective, implementation, and measurements.

Ordinary batch hashes and counters match across all four arms. The three meta
arms share their attack and query data streams. The additional weight-10 arm was
declared before defender training, following gradient calibration: at weight 0.1,
the bounded escape objective contributed a small gradient. The observed weighted
meta-gradient norms range from approximately 0.0096–0.0138 at weight 0.1 and
0.99–18.62 at weight 10, before global gradient clipping. Increasing the weight
does not improve the cheapest observed escape budget. The refusal arm's gradients
have a different magnitude; equal numerical loss weights are not equal update
strengths. The ordinary control does not receive matched extra compute.

A separate shorter-attack search tested learning rates 0.00003, 0.0001, and
0.0003, with saved checkpoints through 1,000 updates. The successful shorter
recipe passed two of three additional attacker data seeds on the ordinary parent.
It was consequently retained as an external alternative rather than silently
replacing the reliable long training attacker. This is an exploratory search,
not a held-out attack-family claim.

## Gradient and checkpoint verification

**46 tests pass**, including copied-model immutability, native two-stage optimizer
equivalence, frozen-displacement finite differences, and exact resumed defense,
attack, and benign-edit training. Resume comparisons cover weights, optimizer,
both data streams, counters, predictions, and numerical history; elapsed-time
fields are excluded from numerical identity. The actual-model first-order
gradient also passes float64 directional checks. See
[`tests.txt`](../../../artifacts/strong-attack/tests.txt) and
[`gradient-check.json`](../../../artifacts/strong-attack/gradient-check.json).

The investigation went further with an experimental differentiable Adam variant
using activation recomputation. Small tests match native Adam values and compare
recomputed versus unrecomputed gradients. The actual 300-step float64 probe
successfully attacks the parent and uses about 4.90 GB of peak process RSS.
It differentiates through every update in that specified stabilized optimizer.
The variance floor is explicit; this is not a claim of bitwise equivalence to
the native float32 300-step attack.

The full derivative is extremely sensitive. Its norm is about **92 million**.
The following checks are all retained:

| Perturbation along normalized gradient | Relative difference from analytic directional derivative |
|---:|---:|
| 1e-10 | 100.57% |
| 1e-11 | 97.47% |
| 1e-12 | 28.35% |
| 1e-13 | 2.50% |
| 1e-14 | 0.42% |

At 1e-14, many intended coordinate changes round away in float64. Accounting for
the *actual representable perturbation* gives 0.087% relative error. Thus this
particular long derivative has numerical support at a very small local scale.
The initial failed checks were not overwritten. A larger Adam epsilon of 1e-4
still produced a successful attack but increased the gradient norm to about
25 billion; it did not eliminate large gradients.

This establishes feasibility of computing and checking a longer derivative,
not a stable or useful optimizer for coupling training. At larger tested
perturbations even the expected descent direction does not consistently reduce
the queried objective. The prototype was kept separate from the validated
first-order campaign. Its gradient files, source copies, initial failures,
smaller-step checks, and representable-direction analysis are under
[`artifacts/strong-attack`](../../../artifacts/strong-attack).

## Cost and evidence

| Defender | Outer updates | Inner updates | Forward token positions in loss evaluations | Local training timer |
|---|---:|---:|---:|---:|
| Ordinary continuation | 256 | 0 | 1,004,736 | 8.60 s |
| Refusal reinforcement | 256 | 20,800 | 73,382,512 | 886.53 s |
| Coupling weight 0.1 | 256 | 20,800 | 73,334,384 | 890.41 s |
| Coupling weight 10 | 256 | 20,800 | 73,334,384 | 866.88 s |

The coupling arms evaluate approximately **73 times** as many forward token
positions as ordinary continuation. These count repeated loss evaluations, not
unique data or equivalent FLOPs. Local timers include contention from concurrent
runs and must not be treated as isolated throughput measurements or GPU pricing.
The 27 registered run directories total 3,557.86 recorded training seconds;
calibration, profiling, numerical checks, evaluation, and checkpoint I/O are
separate. Each successful 300-update shorter attack uses 9,600 batch-example
exposures across its four objectives. No cloud job was launched and GPU credits
spent remain **$0**.

The [audit](../../../artifacts/strong-attack/audit.json) verifies source snapshots,
parent and evaluation hashes, matched ordinary batches and meta streams, and
recomputes training-query scores. The
[larger evaluation receipt](../../../artifacts/strong-attack/additional-evaluation/result.json)
combines immutable evaluation parts with identical data contracts and parent
language scores; their task predictions are independently rescored. Additional
[raw benign outputs](../../../artifacts/strong-attack/benign-predictions/result.json)
verify the benign scores directly. Every run, including unsuccessful attempts,
appears in the [registry](STRONG_ATTACK_REGISTRY.md).

## What this resolves and what it does not

The original three-step training attacker was inadequate. That limitation is
addressed: this experiment trains against long attacks that actually remove the
rule while preserving capability on the parent. It also rules out the default
coupling weight being the only explanation for the negative result, and reveals
a cheaper escape that defeats an apparent benefit under one longer attack.

Neither tested first-order recipe establishes selective coupling. The exact
derivative investigation exposes a severe conditioning problem at the tested
setting, rather than a memory-only obstacle. It does not establish a fundamental
data limitation or impossibility of SCC.

The study still uses one defender initialization, one small architecture, one
permission/table family, and a bounded public-text corpus. Training uses one
long attack profile, while the external shorter profile reveals an unhandled
alternative. Multiple attack profiles during training, better-conditioned outer
optimization, stronger capability tasks, and replication remain research avenues.
There is no evidence here that scaling this recipe would solve those problems.
These limits must stay visible in any review or funding claim.
