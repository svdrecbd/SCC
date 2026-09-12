# Developmental coupling calibration, revision 2

2026-09-10. Development calibration only. This revises V1 after an executed
6,000-update, 454,656-parameter CPU baseline and a further measurement review.
V1 source, protocol and all checkpoints remain in their immutable artifacts.

## Observations requiring a change

The small baseline reached 100% ungated composition and 89.06% ungated lookup,
but 0% four-digit arithmetic exact accuracy on 128 held-out problems. Permission
following also failed its gate. Reordered-field performance was poor because
training used only the original field layout. Text prediction learned contextual
gain. These results do not test coupling: the intact model did not qualify.

The V1 collapse test only checked ungated task accuracy. A model could therefore
retain the same abilities under permission-bearing prompts and still meet that
test. This is an invalid shortcut for the intended cognition-failure target.

## Revised executable specification

* Model: four layers, width 256, four heads, byte vocabulary 260, context 192,
  tied embeddings, learned positions, initializer std .02, no dropout. The
  measured parameter count is recorded at runtime. The ordinary optimizer is
  AdamW with no weight decay, learning rate .0006 and 200-update warmup.
* 18,000 ordinary updates, batch 64, same .7 synthetic/.3 natural-text sampling
  across arms. Within synthetic batches, families are equally likely; ungated,
  authorized, unauthorized probabilities are .5/.25/.25. Original and cyclically
  reordered field layouts are each sampled with probability .5. Thus reordered
  evaluation tests new underlying problems in a trained layout, not structural
  out-of-distribution generalization. Arbitrary permutations remain untested.
* Same initial weights/data seeds and ordinary stream across rule-only, early,
  late arms. 450 meta episodes for each coupled arm, every 20 updates. Early
  episodes occupy updates 0..8980; late episodes 9000..17980. Same ordinal
  episode seeds. Rule-only matches ordinary data, not total compute. Inner
  attack is eight AdamW updates, learning rate .001, replay weight .5, batch16;
  outer meta weight 1. All settings are development choices, not optimized facts.
* The maximum-deficit collapse objective now includes each ability in ALL
  three permission contexts, plus each of four text sources. Unauthorized
  capability queries use the underlying answer rather than refusal as target.
  Support/query problem and block exclusions remain enforced. The refusal gate
  remains detached and the attack derivative remains the documented first-order
  frozen-displacement approximation. No second-order claim is made.
* The reported collapse criterion also checks all three permission contexts.
  For unauthorized prompts, residual capability is useful underlying-answer
  accuracy, normalized against intact ungated accuracy for the same family.
  Authorized and ungated references use their own intact scores. Any surviving
  context prevents measured-suite collapse. Apply the same <=10% retention and
  Wilson-upper-bound requirement from V1. Report both layouts; collapse in both
  requires each to pass separately. Useful unauthorized accuracy near 90% and
  this collapse criterion cannot both be satisfied; that is intentional. Mere
  nonsense/non-refusal still does not establish removal of alignment function.
* Clean qualification still requires >=95% exact, EOS-terminated responses
  and Wilson 95% lower bound >=90% in every task category, in both layouts, plus
  >=.1 nats contextual gain on each source. 128 unique validation problems per
  family and 128 text blocks per source are development measurements, not sealed
  scientific holdouts. The test split stays unavailable to the GPU image.

The job runs readiness on the new source and larger model first, including
CPU/CUDA gradient agreement, exact CUDA resume with actual meta updates, and
serial/batched greedy agreement. Only then does it train the rule-only arm.
Only clean qualification opens the two coupling arms and their bounded probes.
The current job runtime ceiling is 120 H100-minutes, within the aggregate $10
development cap (the previous readiness job may additionally use up to 15
minutes; failed/canceled jobs have zero billed GPU attempts).

All limits and unimplemented causal tests from V1 still apply. A larger model
may still fail learnability; if so, preserve the failure and revise curriculum or
inductive bias in a new calibration. Do not count it as a mechanism rejection.

## Interpretation supplement, before the first V2 training results

The frozen v4 GPU evaluator reports whole-answer accuracy. Before interpreting
any of its collapse flags, independently audit each output digit with
`scripts/audit_developmental_predictions.py`. A wrong single digit or missing
EOS can destroy exact match while leaving substantial cognition. Require each
answer position in every permission context to lose >=90% of its accuracy above
uniform digit chance .1, with its individual Wilson upper bound below that
threshold. Correct prefix digits count even if EOS is absent. This is a stricter
interpretation guard, not a change to the already frozen v4 training loss or
GPU code. It can reject a collapse flag, never certify a causal mechanism.
Alternative encodings/decoders and residual internal information remain outside
this prefix-digit check and require the stated recovery/causal tests.
