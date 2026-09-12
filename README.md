# SCC research

We are trying to build an individual model whose alignment-removing modification
destroys indispensable cognitive computation. **No working SCC mechanism has
been demonstrated.** Useful candidates have been trained and tested; protection
can still be bypassed with substantial abilities retained.

Start with the **[research reset](docs/RESEARCH_RESET.md)**: results, mistakes,
promising and closed directions, assumptions, and the next work.

| Document | Purpose |
|---|---|
| [Research reset](docs/RESEARCH_RESET.md) | Current assessment and decisions |
| [Mechanism target](MECHANISM_TARGET.md) | Stable definition of intended SCC behavior |
| [Operations and repository map](docs/OPERATIONS.md) | Commands, code, evidence and storage |
| [Historical index](docs/archive/README.md) | Archived proposals, workflows and chronologies |
| [Working standards](WORKING_STANDARDS.md) | Evidence, compute and preservation rules |

Latest **observed** job state: 67 of 70 current experiments succeeded and three
memory jobs were running at **2026-09-12 22:46 UTC**. No runtime queries were made
during this reset. Job completion is not scientific success. All eight persistent
models have now been audited and fail ordinary learnability.

The next experiment is a working ordinary learner on the persistent-task
interface, followed by a focused test of what prevents the persistent candidate
from learning. More coupling on an incompetent model would be uninterpretable.

```sh
uv run python scripts/scc_status.py
```

This reads saved exact-batch status. Add `--live` for one explicit API check of
unfinished jobs. There is no automatic watcher.

Scientific code, frozen protocols, run directories and checkpoints retain their
paths. Historical root proposals moved under `docs/archive/`;
[the path map](docs/archive/path-map.json) records their destinations.
