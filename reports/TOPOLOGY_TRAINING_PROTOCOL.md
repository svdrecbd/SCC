# Defender-update diagnostic and comparison — 2026-09-10

This exploratory follow-up is authorized alongside the conceptual review in
[TOPOLOGY_FORMULATION.md](TOPOLOGY_FORMULATION.md). It tests the existing selective
authorization recipe; it does not claim to implement universal self-edit collapse.
Use fresh artifact paths and the same public-text and generated-task distributions.

## Direction diagnostic

From the qualified parent, construct one actual AdamW defender update with the
same ordinary batch and coupling weights 0, 0.1, and 10. Obtain the legacy
first-order gradient from a fresh 1,300-step attack, then compare its predicted
frozen-displacement improvement with newly rerun 1,300- and 300-step attacks.
Use attack seeds 61041 and 61042; only the first long trajectory supplies the
gradient. Ordinary/query seeds are 61881/61991. Evaluate 64 new validation tables
with seed 76351 and the frozen 64-block-per-source language slice. Retain all
predictions and losses. This is a one-update diagnostic, not an estimate of
long-training convergence or an exact gradient through the adapting attacker.

## Matched training comparison

Three arms start from `runs/online-06-byte-mixed/step-00008000.pt`:

- ordinary continuation;
- the legacy escape penalty with alternating long and short inner attacks;
- a per-constraint margin penalty with the same alternating attacks.

All receive 512 ordinary updates, batch size 32, AdamW learning rate 0.0001,
weight decay 0.1, clipping norm 1, seed 271, data seed 2617. Meta updates occur
every eight ordinary updates: 64 episodes, four times the previous count.
The legacy meta arm uses weight 10. The per-constraint arm uses the calibrated
weight described below. Both use batch size eight per attack objective, task query
batches of 32, and eight queried text blocks from each of the four sources.
They share the same sampled query and attack streams. The long profile is the
previous 300-step removal plus 1,000-step recovery; the short profile is 300
steps at learning rate 0.0003 and preservation weight 1. Profiles alternate in
a deterministic order. Total inner updates are 51,200 per meta defender.

Save an initial check at update 8, then evaluate after 128, 256, and 512 outer updates (16, 32, and 64 meta
episodes). Apply fresh external short attacks to these saved models; assess
final models with both short and long profiles and fresh attack data. Use the
same external budgets for all arms. A clean model that loses qualification is
an invalid defense candidate, even if an attack appears to fail afterward.

The ordinary control has no matched extra attack compute. Changing frequency,
duration, query coverage, and attack mix from the earlier campaign is not a
one-factor comparison. The two new meta arms isolate the outer objective while
sharing the new schedule and streams. Preserve all three arms and intermediate
outcomes, rather than reporting only a favorable selected endpoint.

## Per-constraint objective

For each task completion, take its smallest target-vs-best-alternative logit
margin over the supervised sequence including EOS. Positive margins at every
position correspond to greedy correctness absent ties. A sigmoid with
temperature 0.05 gives a differentiable approximation; its mean approximates
the fraction of exact completions. It is not equal to the measured fraction.

The disclosure gate uses threshold 0.90 and temperature 0.05. Authorized and
retrieval gates each use threshold 0.95 and temperature 0.025. Each language
source has its own gate, with threshold equal to the stricter of the current
clean and shared-parent loss on that same training query batch plus `log(1.05)`,
and temperature 0.02. The capability gate is the minimum of these six gates;
the outer penalty multiplies it by the disclosure gate. References are detached.
Record every component and gradient magnitude.

This proxy matches the conjunction of the evaluation constraints more directly
than averaging retrieval and normalized language loss. It is still a surrogate,
can be noisy on small batches, and can be reduced by small capability losses.
It does not implement or measure catastrophic cognition collapse. No claim of
superiority is made before the comparison.

Before defender training, a fixed real-model calibration on the saved seed-10405
short-attack endpoint and query seed 61818 measured gradient norms 0.1677483 for
the legacy loss and 19.3755142 for the new loss. Freeze the new loss weight at
`0.08657747080818477`, matching the legacy weight-10 gradient norm on this one
calibration query (1.677483). This prevents numerical loss scale alone from
setting a 100-fold different initial gradient contribution. It does not match
future gradients, clipping, Adam updates, or compute; report those measurements
separately. The initial unscaled calibration receipt remains intact.

## Verification and decision

Validate the sequence-margin derivative and profile schedule. Exact restart must
preserve the optimizer, profile position, both data streams, and weights in both
objective arms. Test complete real-model greedy outputs rather than treating a
surrogate decline as successful defense. If attacks retain capability after rule
removal, record a counterexample. Failed searches do not certify disconnection.
Use no final test data, imported model weights, or cloud credits.

## Adaptive recovery amendment after the first saved comparison

At outer update 128, both meta defenders remain cleanly qualified. Their short
attacks disclose useful answers after 1,000 updates but fail the text-retention
criterion by producing approximately 8–9% worst-source perplexity growth; the
matched ordinary control escapes. Before interpreting that as a useful barrier,
apply another 1,000 updates of disclosure plus preservation replay at learning
rate 0.0001 and preservation weight 10, with a fresh optimizer and data seed
75441. Evaluate after 100, 300, and 1,000 recovery updates. Apply the same recipe
to all three arms at outer updates 128 and 512, starting from each saved
1,000-update primary short-attack endpoint. Include the preceding 1,000 attack
updates in total escape cost. This is an explicitly adaptive, matched recovery
test, not a held-out attack family selected before observing any results.

## Symmetry-composition amendment

The exact query/key symmetry suggests a coordinate-dependence diagnostic. On all
three final defenders, scale queries by 0.25 or 4 and keys by the reciprocal,
including biases in every attention block. Then apply the same 1,000-step short
attack with data seed 74141, learning rate 0.0003, and preservation weight 1.
Use the unmodified defender as the clean-retention reference. Require the edited
starting model to have identical measured greedy outputs and language scores
before interpreting the attack. This tests two declared rescalings; it is not an
exhaustive search over equivalent parameterizations. Preserve each edited
checkpoint and all six attack outcomes. This adaptive composition is additional
to the original external matrix and recovery amendment.
