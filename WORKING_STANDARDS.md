# Research working standard

The user established on 2026-09-10 that “good enough” is not a stopping rule.
Actively look for concrete improvements in implementation, controls, attacks,
measurement, and interpretation. Execute justified improvements within the
authorized resources. Preserve negative results and explain actual constraints.

Do not represent an untested possibility as an exhausted avenue, a budgeted
search as a proof of impossibility, or a working implementation as a scientific
success. “We cannot make this better” requires evidence and an explicit scope;
it is not a claim of global optimality. Keep the remaining improvements visible
when a particular experiment is complete.

## Primary outcome

The user reiterated on 2026-09-11 that a working SCC mechanism is the primary
objective. A reproducible paper is a secondary deliverable. Completing a
negative-results report does not complete the mechanism-development task.
Preserve and document the current failures while pursuing concrete, testable
changes. Do not promise that the general mechanism is achievable or treat
additional training pressure as evidence that a dependency has formed.

## Current compute authorization

The user's 2026-09-19 placement instruction is: run CPU research jobs on Charon,
even if they take longer. The GMAN grant is GPU-only; reserve GMAN for GPU work.
The user allows an exception for a substantial synchronized CPU batch (for example,
about 12 coordinated jobs) that would occupy Charon all day. Use that exception
only with an explicit workload/concurrency estimate and recorded justification;
ordinary speed advantage is not enough. This concerns compute jobs, not routine
local file editing, Git operations or reading existing artifacts.

The user instructed on 2026-09-11: do not poll runs unless they are especially
short. Submit bounded work, report what was submitted, and let the user report
completion. Keep automatic collectors and scheduled status checks off. Reading
completed artifacts when the user reports back is appropriate. Short implementation
checks may be awaited normally, with CPU jobs placed on Charon under the newer rule.

In the 2026-09-10/11 conversation, the user explicitly removed the previous
aggregate$10 research compute cap and authorized as much compute as the work
needs. That earlier cap is superseded. Continue to choose justified experiments,
record per-job quotes and actual charges, and avoid unnecessary duplicate
artifacts. A chosen per-run wall/output limit remains part of that run's
contract; changing it requires a fresh documented run, not silently rewriting
the old experiment. No additional permission is needed merely because a
justified follow-up exceeds the old aggregate cap.

## Current documentation and status workflow

On 2026-09-19, reiterated 2026-09-20, the user asked to bundle continued research
into substantial work units. Return for a substantial finding, consequential
decision, or before initiating training; keep necessary progress messages concise.
This does not authorize monitoring or weaken experiment admission requirements.

The user's 2026-09-12 local / 2026-09-13 UTC instruction supersedes the reset's
dated-readout workflow. `labnotes.md` is the single living chronological record
and current assessment. Read it before choosing work. Append meaningful plans,
experiments, failures, results, theory decisions, infrastructure changes and
corrections there, with stable entry IDs and evidence links.
Keep entries focused on the question, method, result, specific limits and decision.
Omit repeated acknowledgments, file-reading checklists and unchanged job status;
refer to standing rules instead of restating them. Keep result-specific caveats,
negative results and corrections. The September 20 editorial refresh preserves
an exact original in the evidence store; future corrections remain dated.
Keep one concise current-position block there, with dated observations. Do not duplicate rolling
status in the README, mechanism target or operations guide.

Do not create another narrative report, status, theory or protocol document for
routine work. Describe the next experiment in labnotes before executing it;
declare trigger, editable components, controls, qualification/collapse gates,
repair resources, seeds and budgets. Freeze the exact relevant entry, config
and source in the run's immutable artifact directory. Machine logs, manifests,
receipts and data remain separate files. A distinct human-facing deliverable
is appropriate when the user explicitly requests it.

Keep `MECHANISM_TARGET.md` as the stable definition and `docs/OPERATIONS.md` as
the practical guide. The archived research reset, reports and protocols are
historical evidence, not competing current plans. Preserve their results and
frozen contracts; add dated corrections to labnotes instead of rewriting history.
Historical proposals and early workflows are indexed in `docs/archive/README.md`.

Use `scripts/scc_status.py` to inspect the exact registered experiment batch.
Its default is saved observations only; `--live` is one explicit check, not a
watcher. Account-wide newest-job listings must not be presented as exact-batch
status. Preserve observation dates and separate provider completion, intact
qualification, numerical validation and SCC success.

## Version control

The user requested on 2026-09-14 UTC that research source, tests and documentation
be kept committed and synchronized on `main`. Do not leave completed work only
untracked or on an unnecessary side branch. An isolated checkout may protect an
active experiment, but reconcile its completed changes onto main promptly while
preserving the active run's imported source. Raw data, checkpoints and immutable
run artifacts remain in the separate evidence store.
