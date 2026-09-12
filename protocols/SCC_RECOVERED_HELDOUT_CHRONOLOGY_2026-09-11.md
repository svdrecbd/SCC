# Held-out protocol chronology clarification

The frozen V1 protocol says it was written before collecting the first
four-arm pilot's final scores. That collection clause is too strong: the
background downloader finished its manifest at08:00:33 UTC, while the protocol
file was saved at08:01:16 UTC. The model-visible collector summary was read
after that save, and the first detailed endpoint summary was generated at
08:01:50 UTC. No held-out model scores had been evaluated when the protocol
was fixed. All twelve original arms and the previously selected SEAM parent
were included without outcome-based selection.

The defensible statement is that the held-out design was fixed before test
outcome evaluation. It is a local, time-recorded design, not an independently
registered preregistration. Preserve V1 unchanged with this clarification.
The first held-out job then failed because the cached training context excluded
test files; the replacement supplied their original hash-verified bytes without
changing models, scoring rules, seeds or fitting. No failed-job model scores
were produced.
