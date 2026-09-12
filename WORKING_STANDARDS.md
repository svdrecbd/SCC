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

The user instructed on 2026-09-11: do not poll runs unless they are especially
short. Submit bounded work, report what was submitted, and let the user report
completion. Keep automatic collectors and scheduled status checks off. Reading
completed artifacts when the user reports back is appropriate. Short local
implementation checks may be awaited normally.

In the 2026-09-10/11 conversation, the user explicitly removed the previous
aggregate$10 research compute cap and authorized as much compute as the work
needs. That earlier cap is superseded. Continue to choose justified experiments,
record per-job quotes and actual charges, and avoid unnecessary duplicate
artifacts. A chosen per-run wall/output limit remains part of that run's
contract; changing it requires a fresh documented run, not silently rewriting
the old experiment. No additional permission is needed merely because a
justified follow-up exceeds the old aggregate cap.

## Current documentation and status workflow

The 2026-09-12 reset makes `README.md` the entry point and
`docs/RESEARCH_RESET.md` the current assessment. Read that assessment before
choosing the next research branch. Keep `MECHANISM_TARGET.md` as the stable
definition; do not append run-by-run chronology to it or the README. Save dated
readouts and update the current assessment only when evidence changes decisions.
Historical proposals and early workflows are indexed in `docs/archive/README.md`.

Use `scripts/scc_status.py` to inspect the exact registered experiment batch.
Its default is saved observations only; `--live` is one explicit check, not a
watcher. Account-wide newest-job listings must not be presented as exact-batch
status. Preserve observation dates and separate provider completion, intact
qualification, numerical validation and SCC success.
