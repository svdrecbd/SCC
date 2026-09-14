# Operations

Scientific decisions and observations belong in [labnotes](../labnotes.md).
The [mechanism target](../MECHANISM_TARGET.md) defines what would count as success;
[working standards](../WORKING_STANDARDS.md) cover research and compute rules.

## Setup and tests

```sh
uv sync --extra dev
uv run python -m pytest -q
```

Use the pinned environment through `uv`. The full suite runs on CPU with small
fixtures; GPU experiments have their own numerical and resource gates. To work on
the persistent-task/reference implementation, a focused selection is:

```sh
uv run python -m pytest tests/test_persistent_tasks.py tests/test_persistent_reference.py -q
uv run python -m pytest --collect-only -q
```

The collection command lists individual cases without executing them. Parameterized
cases test different rules, precisions, or boundaries; their count is not a count
of independent experiments or scientific findings.

Tests cover three responsibilities: reusable infrastructure (data separation,
checkpoint/resume behavior, derivatives, and scoring); individual model/runtime
implementations; and regressions that preserve historical counterexamples. Keep
historical tests with the code they validate. Consolidate duplicate assertions or
retire code and its tests together, rather than reducing the count arbitrarily.

## Code navigation

| Area | Entry points |
|---|---|
| Persistent tasks and substrate | [tasks](../scc/persistent_tasks.py), [matrix](../scc/persistent_matrix.py), [runner](../scripts/run_persistent_learnability.py) |
| Ordinary recurrent reference | [GRU](../scc/persistent_reference.py), [runner](../scripts/run_persistent_reference.py), [audit](../scripts/audit_persistent_reference.py) |
| Modification and recovery objectives | [differentiable modification](../scc/differentiable_modify.py), [recovered capability](../scc/recovered_capability.py) |
| Earlier architecture comparisons | [memory factorial](../scc/memory_factorial.py), [portfolio runner](../scripts/run_architecture_portfolio.py) |
| Evidence integrity | [provenance](../scc/provenance.py), [checkpoints](../scc/checkpoint.py) |

Consult labnotes before selecting an experiment. `reports/` and `protocols/` are
historical evidence. Existing runners load named protocols when freezing their
inputs, so those paths remain stable. New plans are recorded in labnotes and
frozen into each new run's artifact directory.

## Compute status

For an evidence-enabled working copy:

```sh
uv run python scripts/scc_status.py
uv run python scripts/scc_status.py --live
```

The default reads saved observations from
`artifacts/developmental-current-status.json`; it does not contact GMAN. A fresh
source clone lacks that ledger. `--live` checks each registered unfinished job
once. Neither command installs a watcher. Do not treat an account-wide newest-job
listing as the status of the registered experiment batch.

Provider completion, numerical validation, task qualification, and evidence for
SCC are separate outcomes. Current compute limits and polling preferences are in
[working standards](../WORKING_STANDARDS.md); dated infrastructure observations are
in labnotes and the [historical archive](archive/README.md).

## Evidence and sharing

Keep datasets, run directories, parent checkpoints, intermediate states, failures,
and frozen source snapshots outside Git and preserve their hashes. Do not edit
source used by an active run. Use fresh artifact paths for new experiments.

A review package needs the relevant labnotes entry, frozen configuration and
source/lockfile, raw predictions, training logs, hashes, and any data/checkpoints
needed to replay the claim. Source alone cannot verify training results. Provider
credentials and signed download URLs are not part of a review package.

Earlier document relocations are recorded in [archive/path-map.json](archive/path-map.json).
The original reset backup and hash ledger remain in the local evidence store at
`artifacts/scc-research-reset-20260912-v1/`. Archived documents and artifact snapshots
must be interpreted at their original dates.
