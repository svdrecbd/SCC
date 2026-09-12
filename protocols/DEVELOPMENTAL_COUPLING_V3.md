# Developmental calibration, revision 3: arithmetic curriculum

2026-09-10. This retains the model, budgets, paired early/late schedules,
full-difficulty validation sets, and interpretation guards of V2. It is a new
calibration, not a revision of any previous run's recorded data or source.

The 3,275,264-parameter V2 model learned composition and improved lookup/rule
following, while arithmetic exact match remained at zero through the observed
13,000-update diagnostic. The next calibration tests whether staged ordinary
training resolves that specific learnability barrier. This is a hypothesis;
neither flat loss nor unsuccessful optimization establishes inability to learn.

The underlying arithmetic problem is unchanged: independent coordinate-wise
addition of three four-digit vectors modulo ten. Only ordinary training's
operand distribution develops with time (zero-based update indices):

* 0..999: B=C=0000, so the model first learns to locate/copy A's digits.
* 1000..4999: C=0000; B's digit range grows from 0..1 to 0..9 in nine stages.
* 5000..8999: B's digits cover 0..9; C's range grows from 0..1 to 0..9.
* 9000 onward: both added vectors use the original full 0..9 digit distribution.

Each generated problem's actual A/B/C values determine its existing hash split.
Queries and permission/rendering counterfactuals do not change that identity.
No held-out example is reassigned to training. Repeated draws are allowed and
counted. A's digits always use 0..9. Earlier phase distributions have fewer
possible problems, a limitation recorded by unique-problem counts.

All three arms use the same curriculum and ordinary stream. Meta support,
meta query, final modification probes, and every validation always use the
FULL-difficulty problem distribution. Early coupling still starts at update0;
late starts at update9000. The different realized difficulty/deficits during
early and late episodes must be reported, not treated as equal effective force
merely because episode counts match. Clean qualification and all collapse guards
are unchanged. A curriculum failure still prevents a meaningful coupling claim.

Run this new calibration only after V2 fails its clean gate. The GPU job runs
numerical/readiness checks first, and must stay within a 60-H100-minute ceiling
and the existing aggregate $10 development cap. Existing model/checkpoint
directories are preserved, including V2. The optional GMAN sample publisher now
records an HTTP401/403 response once and stops those optional calls; stdout and
final artifacts remain the progress/output path.
