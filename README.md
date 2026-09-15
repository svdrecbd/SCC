# Safety-Capability Coupling (SCC)

SCC investigates whether an individual AI model can be built so that removing its
protected alignment machinery destroys cognitive computations it needs to function.

**This is an experimental research program. No working SCC mechanism has been
demonstrated.** The current tasks use synthetic authorization rules as laboratory
proxies for alignment. Passing those tasks, damaging a model, or failing to repair
it within a limited budget does not establish the intended mechanism.

The repository contains PyTorch models, training and modification experiments,
repair attempts, independent result audits, and the research record.

## Read the research

- [Consulting notes — 15 September 2026](CONSULTING_NOTES.md): start here for the
  requested review of the next construction proposal, its failure risks and decision gates.
- [Mechanism target](MECHANISM_TARGET.md): the question, intended endpoint, and
  distinctions that govern the experiments.
- [Labnotes](labnotes.md): current assessment followed by the chronological record
  of plans, results, failures, and corrections.
- [Canonical theory v4.1](deliverables/scc-theory-frontier-20260915/SCC_Theory_and_Editable_Model_Bridge_v4.md): locally reviewed conditional theorem and explicit construction boundary; [derived bridge](deliverables/scc-theory-frontier-20260915/SCC_Editable_Model_Bridge_v4.md).
- [Operations](docs/OPERATIONS.md): setup, code navigation, tests, and evidence.

## Run the checks

The project uses the Python and dependency versions pinned in
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

```sh
uv sync --extra dev
uv run python -m pytest -q
```

The tests use small CPU fixtures and generated data. They do not require GMAN
credentials or the experiment corpus. They check implementation and measurement;
they do not reproduce the full research runs or establish SCC success.

## Repository layout

| Path | Contents |
|---|---|
| [scc/](scc/) | Models, task generators, objectives, interventions, and metrics |
| [scripts/](scripts/) | Experiment runners, numerical checks, result audits, and compute tools |
| [tests/](tests/) | Implementation checks and regressions for current and historical experiments |
| [labnotes.md](labnotes.md) | The single living research record |
| [reports/](reports/), [protocols/](protocols/) | Historical findings and frozen experiment contracts |
| [docs/archive/](docs/archive/) | Original proposals and superseded workflows |

Historical material describes the work at its recorded date. It is not a second
set of current instructions. Contributor instructions are in [AGENTS.md](AGENTS.md)
and [WORKING_STANDARDS.md](WORKING_STANDARDS.md).

## Evidence and reproducibility

Datasets, checkpoints, and run artifacts are excluded from Git. Links into
`artifacts/` and `runs/` in the labnotes require the corresponding evidence package;
they will not resolve in a fresh clone. Reproducing a reported result requires its
frozen source, configuration, data, checkpoints where needed, and raw predictions.
See [operations](docs/OPERATIONS.md#evidence-and-sharing) for the evidence workflow.
