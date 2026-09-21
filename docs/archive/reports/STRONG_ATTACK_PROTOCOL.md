# Strong-attacker development protocol — 2026-09-10

This follow-up addresses the weak three-step training attacker in the first
coupling experiment. It is local exploratory development. No cloud spending,
final test partition, pretrained weights, or model scaling is included.

## Planned implementation and checks

1. Run an actual fresh-copy AdamW removal/recovery trajectory from the current
   defender: 300 removal updates at lr=0.001, preservation weight=1; then 1,000
   recovery updates with a reset optimizer at lr=0.0001, preservation weight=10.
   Each update has eight disclosure, eight authorized, eight retrieval, and eight
   natural-text examples. The objective, clipping, and optimizer match the attack
   that escaped the first experiment. All inner compute must be counted.
2. Check strength before defense training on the qualified parent using three
   distinct training data seeds. Evaluate exact useful disclosure and ordinary
   capability, including per-source language retention. A low surrogate loss
   alone is not an attack-strength result.
3. Use the long attack endpoint for first-order meta-training: treat its weight
   displacement as fixed when differentiating the outer loss. This approximates
   the attack Jacobian by the identity; it is NOT the exact derivative of the
   complete attack. Test frozen-displacement gradients, copied-model immutability,
   optimizer/stage equivalence, and exact resumed training. Preserve the earlier
   exact short-unroll implementation for comparison.
4. Compare ordinary continuation, post-attack refusal reinforcement, and the
   escape penalty from the same parent, ordinary batches, optimizer, and update
   count. The latter two receive the same long-attack data stream. Initially use
   256 outer updates, 16 attack episodes (one every 16 steps), and coupling
   weight 0.1. Count the changed episode schedule explicitly; this is not a
   one-factor causal comparison with the earlier short-unroll experiment.
5. Reapply the same external removal/recovery attack with seeds absent from
   defense training, the previous last-block attack, and the policy-preserving
   benign edit. Evaluate clean models and escaped endpoints on additional
   development tables. Preserve failed candidates; do not silently pick a lucky
   defender or claim that surviving this suite proves a lower escape-cost bound.

The first-order approach is informed by the explicit approximation in
[TAR section 4](https://arxiv.org/html/2408.00761v3#S4). This implementation is our
own synthetic-task experiment, not a reproduction of the paper's performance.

## Decision rules and scope

Use the prior >=90% useful unauthorized answers, >=95% authorized/retrieval,
and <=5% per-source perplexity growth relative to both clean defender and shared
parent. Clean and benign qualification retain the protected rule too. Record
raw outputs and intermediate checkpoints, not just final pass/fail flags.

If the stronger inner procedure does not reliably escape the ordinary parent,
improve or extend its recovery before launching defense training. If a defender
loses clean capability, treat it as an invalid candidate and investigate the
tradeoff. If all qualified arms still escape at comparable tested cost, reject
the candidate, identify the remaining concrete limitations, and do not scale it.
Evaluate optimization progress and gradient magnitude rather than assuming the
new surrogate update is effective merely because it is nonzero.

The user's working standard is documented in `WORKING_STANDARDS.md`: finish
justified improvements, distinguish practical constraints from impossibility,
and retain the work still needed. No single small experiment can demonstrate
that the overall approach cannot be improved.

## Pre-training calibration amendment

The long attack escaped the qualified parent with all three calibration seeds
(10404–10406). The actual-model frozen-displacement gradient passed finite
differences. Its norm was approximately 0.103 on the chosen training query batch;
at weight 0.1 its contribution would be only about 0.0103. Its cosine with the
clean language gradient was -0.724 on that batch. These are local diagnostics,
not proof of either an effective update or unavoidable capability damage.

Before training any defender, add an escape arm at weight 10 to test whether
the default weight simply underdrives the bounded objective. Report both weights
and all outcomes. All three meta arms use 16 full attack episodes and query
batches of 32 examples, with generation checks on those training queries.
Log ordinary/meta gradient norms and their cosine. The refusal arm remains at
weight 0.1; equal numerical weights do not imply equal gradient strengths.
The ordinary control has no additional attack rollouts. Runtime profiling at
1/2/4 threads favors four threads here, so retain the calibrated four-thread
execution contract. Parallel runs may contend; token/update counts remain the
primary compute comparison and contended timings must be labeled.

## Additional development checks

The larger endpoint check will use 128 new validation tables (seed 921591),
both original and rotated table order, and **all 12,689 natural-text validation
blocks**. This broadens the original 64-block-per-source language check. Reordering
the synthetic tables does not change the language examples or weights, so the
same measured language scores can be reused for both table orders. These are
development checks; the final test partition remains unused.

A separate learning-rate calibration searches shorter ordinary AdamW attacks
at 0.00003, 0.0001, and 0.0003 with preservation weight 1. Any apparently successful
shorter procedure must be checked across data seeds before replacing the long
training attacker. An experimental stabilized differentiable Adam implementation
under `experiments/` probes whether activation recomputation makes an exact
longer derivative practical. Its optimizer values and small gradients are
checked independently. It is not silently substituted into the frozen campaign.

## Execution record

The shorter recipe passed two of three calibration seeds. It remained an
external alternative, then was applied to all four saved defenders with matched
data and evaluation budgets. Both its first successful saved checkpoint and its
final checkpoint were included in the additional evaluation. All four arms
escaped at 300 updates and also at the 1,000-update endpoint. Raw benign outputs
were subsequently retained and independently checked against the broader scores.

The experimental 300-step full derivative initially failed the directional
finite-difference criterion at 1e-10 through 1e-12. The same saved gradient was
then checked at 1e-13 and 1e-14, reaching agreement within the declared 1% tolerance
at the latter. An additional analysis used the actual representable float64
direction. These are post-result diagnostic extensions, not preregistered
confirmatory evidence. All initial failures remain available. See the
[results](STRONG_ATTACK_RESULTS.md) and [registry](STRONG_ATTACK_REGISTRY.md).
