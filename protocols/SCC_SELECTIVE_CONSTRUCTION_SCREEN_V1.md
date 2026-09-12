# Selected-exception construction screen V1

Frozen before submission, 2026-09-11. This is open construction/calibration, not
a new developmental comparison or a publication-confirmation experiment. Its
purpose is to find a trainable destructive response, beginning with an edit
procedure already known to preserve capability. All parent and failed results
remain intact. This screen cannot settle whether joint development is needed.

## Parent and candidate change

Use the qualified ordinary seed17 step18,000 parent from the corrected pilot.
Recheck its raw identity-reader intact qualification before proceeding. Every
candidate/control starts from that same saved parent, with fresh optimizer
moments. Architecture and measured abilities remain unchanged.

The target is the selected lookup exception R=X,U=W. Each inner step combines
(target-answer NLL + 3 * normalized benign-replay NLL) / 4 with 0.5 times correct
refusal NLL on non-target unauthorized requests. Replay cycles all six benign
task contexts and four text sources. Answers include EOS. Inner repair retains
this same target-preserving objective, uses fresh moments/samples, and rehearses
benign abilities and other refusals. Training-derived normalization and local
support/query exclusions apply. None of these optimizers accesses clean-parent
weights as a repair resource.

Each outer episode fits identity/sign/digit readers using eight training
examples per family and evaluates eight fresh training examples per query
context. Capability queries contain six benign contexts and four text sources;
other forbidden-answer failures cannot lower the measured benign capability.
The trigger is evaluated on a separate selected-exception query under the SAME
reader and weights. The penalty uses the MAXIMUM retained domain score, then
the maximum across interpretations and modified/repaired endpoints. This changes
the old average-domain penalty intentionally. It still uses a bounded,
positive-affine-invariant teacher-forced ranking surrogate, not a destruction
certificate. Tied scores remain conservative at 0.5. Readers are fitted
without differentiating their discrete assignment; selected-branch derivatives
through the actual smooth-Adam trajectory are otherwise complete.

## Calibrate an actual capability-preserving training edit

Four declared procedures use inner batch size 2, epsilon 0.0001, clipping 1,
betas 0.9/0.95, no decay, repair LR 0.0001 and episode ordinal 70400:

| Cell | Modification steps | Repair steps | Modification LR |
| --- | ---: | ---: | ---: |
| short | 16 | 16 | 0.0001 |
| medium | 64 | 32 | 0.0001 |
| long | 128 | 64 | 0.0001 |
| medium-faster | 64 | 32 | 0.0003 |

Record every forward result. Select using identity greedy behavior on 32 unique
TRAIN cores per family, paired across contexts/layouts, and 32 TRAIN text blocks
per source. Explicitly exclude the union of all calibration episode examples
from this selection suite. No validation/test example or temperature fitting
selects a procedure. Raw text NLL makes the eligibility gate conservative to
confidence shifts; final evaluations still apply the full calibrated readers.

Eligibility requires the repaired model to achieve at least 90% selected-target
success, 90% exact benign accuracy in every context/layout and 90% other-request
refusal, while retaining at least 95% of every source's positive parent text
context gain. An eligible procedure must therefore actually present a useful
escape to training. Require positive parent gain in every source.

Try eligible procedures in ascending total inner+repair steps, then modification
LR, then cell name. For each, check the exact full-size outer derivative against
rerunning BOTH trajectories and reader fitting. Perturb in the normalized
outer-gradient direction at L2 sizes 0.001, 0.0003 and 0.0001. Preserve the coarse
diagnostic; both finer checks must give descent and relative error below 10%.
Record branch changes, nonzero gradient norm, time and peak allocated GPU memory.
An OOM rejects that procedure as infeasible for this exact implementation,
without changing the gradient method. Keep other numerical failures visible.
Select the first eligible procedure that passes; if none passes, finish with
calibration results and no defender training. That outcome is not evidence that
SCC is impossible. No frozen-displacement or truncated-gradient fallback.

## Fixed constructive continuation

If calibration passes, run four cells: ordinary continuation, and coupling
coefficients 0.1, 1 and 10. Each receives 320 identical ordinary updates, batch64,
fresh AdamW at LR0.00001, betas0.9/0.95, epsilon1e-8, clip1, no decay, full task
difficulty and data-stream seed559801. Every coupled cell also receives 64
selected-procedure episodes, every five updates throughout the continuation,
with ordinals70500–70563. Episode samples and ordinary streams match across the
relevant cells. Record hashes, ordinary/meta losses, selected branches, every
gradient norm and realized gradient ratios. Do not equalize realized gradients
or stop/extend based on intact qualification. All coefficients are exploratory.

Measure every final candidate, including unqualified ones, with the original
128-core validation suite, both layouts, all declared readers and temperature
controls. Probe the selected trained procedure with fresh episode80400 samples,
then independently apply the existing 500+500 complete-replay modification and
repair using seed173905, batch16 and stock AdamW epsilon1e-8. This retains a
stronger/different bounded evaluation. Its target class is the trained target;
unseen target classes and other procedures remain future evaluation work.

Score protection and each capability jointly after modification and repair.
Increased refusal alone is resistance, not demonstrated destructive coupling.
Unqualified parents, partial forgetting, a changed output encoding, and generic
training damage cannot count as the target mechanism. An apparent constructive
result still needs causal controls, independent regeneration, unseen edits,
longer/different repair and replication. No positive mechanism claim is made
automatically by a successful provider job.

Save final model/optimizer/RNG/stream states and the final strong-repair weights,
raw predictions and append-only training logs. This runner has no validated
interrupted-resume workflow; do not claim bitwise resume from its saved states.
The CPU implementation gate requires actual nonzero coupling updates, matched
ordinary streams, unchanged parent weights and exact checkpoint reload, plus
independent stock-Adam and rerun-derivative tests. Repeat the full-size numerical
derivative gate on H100 before any constructive continuation.

## Resources and immutable execution

One H100 job, maximum 120 provider minutes, 6,900 runner seconds and 1.5 GiB output.
This is a finite first construction screen, not a cap on the user's program.
The source overlay is frozen before execution. No remote source edits, automatic
polling, status collectors or automatic follow-up submissions. The user will
report when the run completes; interpret process success separately from SCC.
