# Naming standard

Every name you choose, for files, scripts, directories, functions, variables,
classes, commits, branches, anything, must use precise professional vocabulary.
Choose the word a 1972 IBM engineer would write in a specification. No slang, no
casual shorthand, no cute or clever names, no words borrowed from chat culture.
This applies universally, not only to the one example that follows: a script that
deploys dashboards is `deploy_dashboards.sh`, not `push_dashboards.sh`. That is one
illustration of the principle, not the extent of it.

# Typography standard

Use Palatino throughout human-facing documents: body text, titles, headings,
bylines, tables, captions and figure labels, unless the user specifies otherwise.
Use a compatible Palatino-family mathematics font where specialized mathematical
symbols require it. Keep historical artifacts unchanged unless a revision is requested.

# Project working instructions

Read `labnotes.md` before research work or status interpretation. It is the
single living chronological record and current assessment. Append meaningful
plans, experiments, results, failures, theory decisions and corrections there;
update its current-position block when needed. Do not create new narrative
reports, status documents or theory/protocol documents for routine updates.
Freeze the relevant labnotes entry, machine configuration and source in each
new run's artifact directory. Existing reports/protocols remain historical
evidence. Separate human-facing deliverables require an explicit user request.

Read `WORKING_STANDARDS.md` before research changes. The user expects concrete
improvements to be pursued, with actual limitations stated rather than a
“good enough” stopping rule.

Read `MECHANISM_TARGET.md` before research changes or interpreting results. It
records the user's destructive cognition–alignment coupling goal. Keep the
synthetic authorization proxy and utility-retention thresholds distinct from
the intended mechanism and catastrophic cognition-failure endpoint.

Preserve prior experiment directories, failures, source snapshots, and parent
checkpoints. Use fresh paths for new runs. Distinguish development calibration,
implementation validation, and scientific evidence. Report approximation and
resource limits explicitly; never convert unsuccessful search into a proof.

Do not edit imported source while a multi-run process is executing and taking
new source snapshots. Finish or stop that process before changing its source.
