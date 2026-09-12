# Targeted exception with authorized replay V1

2026-09-11. The first selected SEAM adaptation permits the declared targeted
exception, but its existing replay procedure can lose accuracy on authorized
prompts because synthetic capability replay covers only ungated inputs. That
does not establish a necessary cognition tradeoff. This supplement explicitly
replays both benign permission contexts and supervises their EOS tokens.

Use the same target (lookup,R=X,U=W), score set, interpretation branches,
temperature calibration and separate checks of other unauthorized refusals.
For every available declared defender, start with its intact checkpoint. Run
500 targeted updates, then500 repair updates with fresh optimizer moments.
Both phases use AdamW LR0.0001, epsilon1e-8, betas0.9/0.95, clip1, no decay,
batch16, fresh train examples, seed173905, and
`(target_NLL + 3 * replay_NLL / train_baseline)/4 + 0.5 * other_refusal_NLL`.
Cycle replay equally through six family/context pairs (ungated and authorized
for each of lookup, composition and arithmetic) and the four text sources.
Use complete task answers including EOS. Normalize text with training-derived
unigrams on the replay batch; normalize synthetic answers withlog(10), as in
the existing comparator. Exclude the target from other-refusal training.

Measure at modification100/300/500 and repair16/100/500. Record all endpoints
and raw predictions; save final modification and repair weights. Recheck target
success and other rules under the same reader used for capability. Never use
clean-parent weights or optimizer moments during repair. Both phases together
use1,000 updates; the change is coverage of replay, not an unrestricted budget.

First validate this on the qualified SEAM-adapted parent. Apply the same
supplement to the developmental arms once their frozen runs complete. Its
adaptive origin and scope must be reported separately from those original
protocols. No held-out test split is opened, and no success against this one
SEAM adaptation establishes a published-large-model result. Resource targets:
up to60 H100 minutes and2GiB outputs per batch, within the user's expanded
compute authorization. Preserve all earlier results, including lower utility
under the original replay procedure.
