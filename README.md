# SCC research

We are trying to build an individual model whose alignment-removing modification
destroys indispensable cognitive computation. **No working SCC mechanism has
been demonstrated.**

Start with **[labnotes.md](labnotes.md)**. It is the single living record:
current position first, then the chronological experiments, results, mistakes,
theory, decisions and evidence. Future research updates go there.

| Reference | Purpose |
|---|---|
| [Labnotes](labnotes.md) | Current assessment and complete documented research chronology |
| [Mechanism target](MECHANISM_TARGET.md) | Stable definition of intended SCC behavior |
| [Operations](docs/OPERATIONS.md) | Commands, code, evidence and storage procedures |
| [Working standards](WORKING_STANDARDS.md) | Research, documentation, compute and preservation rules |
| [Historical index](docs/archive/README.md) | Earlier proposals and archived workflows |

```sh
uv run python scripts/scc_status.py
```

This reads saved exact-batch observations. Add `--live` for one explicit API
check of unfinished jobs. There is no automatic watcher. Scientific code,
historical reports/protocols, run directories and checkpoints retain their paths;
they support the labnotes rather than compete with them as current accounts.
