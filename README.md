# Safety–Capability Coupling (SCC)

Can an individual AI model be built so that removing its protected alignment
machinery destroys the cognitive computations it needs to function?

**This is an experimental research program. No working SCC mechanism has been
demonstrated, and the negative results do not establish general impossibility.**
Synthetic authorization tasks are laboratory proxies, not the intended endpoint.

## Start here

1. **[Safety–Capability Coupling Program](output/pdf/Safety_Capability_Coupling_Whitepaper.pdf)** — canonical two-column whitepaper, version 1.3: the research question,
   central mathematical lessons and next direction. [Editable source](deliverables/scc-whitepaper/Safety_Capability_Coupling_Whitepaper.md).
   [Detailed technical reference](docs/archive/whitepaper-v1.1/README.md).
2. **[Labnotes: current position](labnotes.md#current-position)** — assessment,
   research gate and complete chronology, grouped into expandable phases.
3. **[Mechanism target](MECHANISM_TARGET.md)** — what the project is trying to
   establish and which proxy outcomes do not establish it.
4. **[Operations](docs/OPERATIONS.md)** — setup, code navigation, compute and evidence.

[Historical archive](docs/archive/README.md) contains earlier reports, proposals,
consultations and budget plans. [Dated deliverables](deliverables/README.md) include
the theory package and figures. Read these at their recorded dates; the labnotes
contain subsequent corrections and the current assessment.

## Work with the code

| Path | Purpose |
|---|---|
| [scc/](scc/) | Models, tasks, interventions and metrics |
| [scripts/](scripts/) | Runners, independent audits and maintenance tools |
| [experiments/](experiments/README.md) | Finite construction, recovery and proof prototypes |
| [tests/](tests/) | Implementation checks and historical regressions |
| [configs/](configs/) | Versioned experiment configurations |
| [protocols/](protocols/README.md) | Frozen historical contracts retained for older runners |

Use the pinned environment in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).
Run CPU research/test jobs on Charon under the [compute rules](WORKING_STANDARDS.md):

```sh
uv sync --extra dev
uv run python -m pytest -q
```

These checks validate implementation and measurement; they do not reproduce the
full experiment corpus or establish SCC. Contributor instructions are in
[AGENTS.md](AGENTS.md) and [WORKING_STANDARDS.md](WORKING_STANDARDS.md).

## Sharing and evidence

The share ZIP contains committed source/documentation and explicitly listed small
evidence supplements. `_SHARE_INFO.json` records the commit, scope and file hashes.
It excludes private notes, credentials, environments, bulk data and checkpoints.
See [packaging instructions](docs/OPERATIONS.md#refresh-the-share-zip).

Most `artifacts/` and `runs/` links need the separate evidence store and will not
resolve in a source clone. Reproducing a result requires its frozen plan, source,
configuration, raw outputs and any required data/checkpoints. All original failures
and source snapshots remain preserved.
