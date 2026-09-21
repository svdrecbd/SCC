# SCC: a funding-oriented pilot within $450

Date: 2026-09-09. Proposed protocol and spending allocation; no experiments have
been run and no result is implied. The existing research ladder remains the
long-term plan. This proposal targets preliminary evidence below its Tier-1 scale.

## The decision this pilot should inform

Does deliberately coupling a protected rule during representation formation
produce a repeatable advantage over invariant-only training and a matched late
coupling schedule, when attackers explicitly preserve and recover utility?

The desired deliverable is a bounded, independently inspectable reason to fund
the next experiment. It is not evidence of safety in modern LLMs, a scaling law,
an economic guarantee, or proof that no escape exists.

This positioning matters because SEAM already studies capability collapse under
harmful fine-tuning. The potentially informative difference is training history
and robustness to adaptive recovery, not the existence of collapse alone.

Sources:
- [Self-Destructive Language Models / SEAM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1abb0e7bd62ba80610798dee81950522-Abstract-Conference.html)
- [One Step to the Side](https://arxiv.org/html/2605.14605v1), a preprint reporting
  capability-preserving attacks on several existing defenses. Its findings do not
  establish impossibility for SCC.

## Primary experiment

Target one conventional decoder-only Transformer of approximately **50M total
parameters**, including embeddings. Target **1B outer-training tokens per run**,
with short contexts suitable for the chosen tasks. These are provisional sizes:
actual throughput, memory, and task learnability must be checked in the first
allocation. Use a single H100 unless profiling demonstrates a reason to change.

All model lineages start from our random initialization. Data preparation,
tokenizer training, configuration, and development are under our versioned
pipeline. Open-source libraries and public corpora can be used with pinned
versions, licenses, and content manifests. No external pretrained weights are
needed for the primary arms.

Train three primary arms across **three paired initialization seeds**:

| Arm | Protected-rule training | Coupling treatment |
|---|---|---|
| A: invariant-only | From the start | None |
| B: early coupling | From the start | Episodes distributed through training, beginning early |
| C: late coupling | From the start | Same total coupling-episode budget concentrated late |

The primary comparison is B versus C; A establishes ordinary removability.
A single additional ordinary-LM reference can calibrate clean utility costs. This
gives **nine primary final checkpoints plus one calibration checkpoint**. The
single ordinary-LM reference cannot support seed-robust claims of LM parity.

Hold architecture, outer tokens, data mixture and order, tokenizer, optimizer,
and evaluation constant within a seed. B and C must have the same number of
coupling episodes, inner update counts, and auxiliary example exposures; vary
their timing. Record total compute. This tests the temporal placement of coupling
at matched dose. It does not by itself test every developmental hypothesis or
show that the rule must be introduced during pretraining.

Use one affordable, explicit coupling objective with short inner attacks. Prefer
an attack that already includes capability preservation. Treat gradient-overlap
measurements as secondary diagnostics. Freeze the objective, schedule, data
mixture, thresholds, and attack protocol after the discovery allocation, before
running the three final seeds. Do not choose final settings using those seeds.

## Candidate protected rule and independent utility

Use a synthetic authorization task with exact ground truth. Each example includes
randomly generated records, permissions, and a query. Authorized queries require
retrieval or a short composition over the records; unauthorized queries require
withholding the answer. Randomize names, record contents, task variants, and
surface forms. Split latent problem instances before rendering templates.

The same underlying computations must be solved successfully on authorized
instances before interpreting any protection result. A blanket refusal model
does not pass. The rule is synthetic; there is no claim of real authentication
or of keeping a key secret from a white-box attacker.

Separate three measurements:
1. Correct authorized task execution.
2. Unauthorized disclosure of the **correct underlying answer**.
3. Utility on distributions without the protected output convention.

An unauthorized nonsense answer may violate the output rule, but it does not
demonstrate useful attacker success. Report both rule violation and useful
unauthorized performance. Internal retention of the rule does not excuse a
successful behavioral bypass in a modified model.

Independent utility should include held-out natural-language modeling and a few
learnable synthetic families such as arithmetic, compositional transformations,
and retrieval. Score their answers without requiring the protected format or
authorization marker. Report every domain and authorized-task utility separately;
do not hide preserved attacker-useful competence behind a falling average.

This candidate is deliberately vulnerable to an obvious objection: the model
may learn the computation and authorization decision separately. Test final-layer
or conditional overrides early. If they cheaply defeat it, redesign within the
discovery allocation or report that this candidate failed. Do not spend the
remaining funds replicating an already falsified claim.

## Attacks and budget discipline

Use matched attack-search effort with a small, predeclared grid for each arm:

- Full-parameter removal with capability replay/preservation and a sweep over
  its weight; compare the best observed escape, not a single destructive setting.
- Low-rank or final-layer adaptation that attempts to bypass the rule while
  retaining the useful representation. Hold one adaptation family out of defense
  development and training.
- Two-stage removal and recovery, including continuation from temporary utility
  collapse. Count both stages in total attack cost.

Reserve the strongest final attack test until the defense is frozen. Use separate
attack-development and final test instances. All arms get the same protocol and
search allocation, but attackers may select different effective hyperparameters.
Attacks need not produce monotone trajectories or use the defender's optimizer.

Include at least two benign edits of comparable measured difficulty on A, for
example learning a new harmless mapping and adapting to a benign data domain.
Measure whether the intended edit succeeds, as well as collateral utility loss.

Distinguish cost to execute the cheapest observed successful attack from total
search cost including unsuccessful trials. Record actual billed runtime, training
tokens, and estimated FLOPs. No observed escape within B is a search result, not
a mathematical lower bound on the minimum possible escape cost.

## What would be encouraging enough to seek funding?

The following are proposed practical targets to finalize after smoke tests and
before final seeds, not promised outcomes or universal scientific thresholds:

- All three primary arms learn the rule reliably; a starting target is at least
  95% rule compliance, alongside nontrivial authorized-answer accuracy.
- Clean utility is closely matched across A/B/C. Specify a margin for each
  metric, such as 3 percentage points on task accuracy and 5% relative perplexity;
  do not apply an accuracy margin to perplexity or chance-floor scores blindly.
- Attacks can make A produce useful unauthorized answers while retaining utility.
  Otherwise the attacker or the task has not been validated.
- B has a meaningfully better observed attack/utility frontier than A and C across
  the three seeds, with a held-out attack family supporting the same direction.
- At comparable successful rule degradation, some attacks on B produce greater
  loss of independent utility. If the rule never breaks, characterize the finding
  as tamper resistance rather than demonstrated collapse coupling.
- A useful ambition is a **3–5× increase in the cheapest observed escape cost**,
  or no escape found under a predeclared multiple of a control-calibrated budget.
  A small cost multiplier is preliminary evidence of mechanism, not a claim of
  economic deterrence. Apply a fixed calibration rule without examining final
  test instances, and enforce the global attack-spending limit.
- Capability-preserving and recovery attacks fail to cheaply erase the advantage,
  while matched benign edits remain achievable without comparable utility loss.

Three seeds give an initial replication check, not a precise population estimate.
Show all seed results and label uncertainty correctly. Thousands of test examples
do not become thousands of independently trained models. Do not assert a decisive
effect merely because an example-level significance test is small.

If B beats A but not C, the result supports a coupling reproduction but not the
developmental hypothesis. If the advantage vanishes under recovery or final-layer
adaptation, report that failure. If all edits damage utility, report brittleness.
If controls never learn the task well, the pilot is inconclusive about coupling.
These outcomes should not be packaged as positive evidence for the original claim.

## Spending allocation

Normal batch rate: $2.997 per H100 GPU-hour, checked 2026-09-09 at
[GiveMeANode pricing](https://givemeanode.com/). Discounts are upside; the plan
does not depend on them.

| Work | Maximum planned allocation | Approximate batch GPU-hours |
|---|---:|---:|
| Small smoke tests, learnability, throughput, memory, resume and obvious bypasses | $30 | 10 |
| Nine primary training runs and one ordinary calibration run | $180 | 60 |
| Adaptive attacks, held-out evaluation, recovery, and benign edits | $150 | 50 |
| Reserve for reruns, build/data/storage charges, and unresolved checks | $90 | 30 before non-GPU charges |
| Total | **$450** | **150 before non-GPU charges** |

Feasibility arithmetic for 50M parameters and 1B outer tokens:

    ordinary FLOPs ≈ 6 × 50e6 × 1e9 = 3e17
    ordinary cost ≈ $1.665–4.995 at 150–50 effective TFLOP/s
    primary + calibration cost = ordinary cost × [3 × (1 + 2m) + 1]
    where m = measured SCC/ordinary training cost at matched outer tokens

For m between 2 and 5, the training block is approximately **$27–170** across
those throughput scenarios. This makes $180 a plausible training allocation,
subject to actual measurement. It excludes the separate discovery and attack
allocations. Auxiliary coupling compute is in m and must not be silently omitted.

Proceed with the final campaign only after a timed sample, including overhead,
projects it within the training allocation and the task is learnable. If necessary,
choose a smaller size or token count before freezing the design. Preserve
replication and attack evaluation ahead of increasing parameter count. If a
learnable design cannot fit, finish with the measured feasibility result rather
than exceed the budget or imply a positive result.

The CLI's previously observed $53.88 monthly workspace cap is separate from
reported credits and still needs reconciling before execution. This proposal does
not change it or authorize spending. Local preparation can proceed independently.

## The funding package

1. A concise report stating the hypothesis, experimental design, all results,
   competing explanations, limitations, and the exact next funded experiment.
2. Three main figures: best observed utility versus useful rule violation;
   attack/recovery success versus total compute; benign edit success and utility
   retention. Show each seed and distinguish early from late coupling.
3. Checkpoints, code/configuration hashes, data and tokenizer manifests, seeds,
   resumable training state where practical, raw evaluations, and attack logs.
4. A single-command regeneration of the reported metrics and figures from saved
   artifacts, plus instructions to reproduce training under a declared budget.
5. Measured training/attack throughput and an updated costed proposal for a
   125–350M replication, a second invariant, and stronger independent attacks.

If supported by results, the funding statement can say:

> In three independently seeded small Transformers trained from random
> initialization, early coupling improved the observed modification-cost/utility
> tradeoff relative to invariant-only training and a matched late treatment.
> The effect persisted under the specified held-out and recovery tests, while
> benign adaptation remained possible. We seek funding to test whether it
> survives larger models and a second protected rule.

Only include clauses established by the actual data. A successful pilot would
justify testing scalability; it would not establish scalability already.
