# SCC mechanism: shared producer and editable reader

The next candidate is implemented, trained and independently audited. Sharing
the complete learned reader closes the previous separate-polarity/offset bypass:
compensating the shared computation restores permission as well as task behavior.
However, **the intended SCC mechanism is still not demonstrated**.

The sparse-permission candidate has an in-bound removal procedure retaining
99.41–100% task accuracy after recovery. Making permission calls identical to
memory calls creates a narrower behavioral tradeoff, but its most dramatic task
failures still preserve information recoverable with a calibrated readout.
The distinction between those two outcomes is central to interpreting this run.

## What changed

The prior construction used a learned comparison followed by a specified
memory reader. The new model includes the comparison, its shared affine
calibration, and a neural reader as editable parameters from the start.
The shared model has 7,556 parameters. The reader has two 32-unit tanh layers
and produces one bit logit from weighted payload sum, total match weight and
payload cardinality. Permission and cognitive calls use that same reader.
There is no cognitive-only interpretation parameter in the shared model.

Two permission distributions use this architecture:

* **Sparse:** a single permitted identity is represented by a one-hot table.
  This retains the preceding equality-permission proxy.
* **Matched:** an access-control table can grant arbitrary identities; reading
  its permission bit uses the same data distribution as cognitive bit lookup.
  Evaluation also crosses the actual lookup bit calls with the permission role.

This changes the permission proxy as well as its distribution. Benefits in the
matched arm cannot be attributed solely to a better architecture. A third
condition copies each qualified matched parent and gives permission its own
reader. That uncoupled-reader control has 8,773 parameters and tests the effect
of permitting a separate interpretation.

Three preserved predicate initializations were reused, with verified hashes.
Six shared models were trained for 1,000 updates each, plus three copied
controls. All nine conditions achieved 100% clean task accuracy and correct
primary permission decisions. Five edit procedures ran for 600 updates each;
a selected endpoint received another 1,000 recovery updates with fresh Adam
moments. Reader-only edits and joint edits were both tested, using matched
streams across corresponding runs.

These remain symbolic memory experiments: byte retrieval, two-table composition
and addition of retrieved bytes. The task controllers and bit assembly are
specified in code. The reader receives no caller-role label. Keeping that
interface fixed and withholding the wider calling context are assumptions,
not demonstrated properties of a general-purpose model.

Protocol: [SCC_SHARED_READER_V1.md](../../../protocols/SCC_SHARED_READER_V1.md).

## Removal and recovery results

Each endpoint was evaluated on the same 12,288 fresh memory problems: 2,048 per
task in each of independent and balanced payload conditions. There are 256
sparse permission pairs, and 16,384 paired permission-bit decisions per lookup
payload condition. Rates below span all three initializations and six task
domains. The primary permission condition differs between sparse and matched.

| Model / edit | Unauthorized acceptance | Exact task accuracy |
|---|---:|---:|
| Sparse, reader-only edit | 100% | 98.10–100% |
| Sparse, joint edit with high cognitive replay | 100% | 98.10–100% |
| Sparse, joint recovery | 100% | 99.41–100% |
| Matched, reader-only edit | 16.00–31.38% | 18.26–100% |
| Matched, joint edit with low replay | 100% | 0.20–0.54% |
| Matched, joint edit with high replay | 0% | 100% |
| Matched, joint recovery | 0% | 99.90–100% |
| Separate reader, reader-only edit | 100% | 100% |
| Separate reader, joint recovery | 100% | 100% |

Primary legitimate acceptance is 100% in all of these listed endpoints. In the
sparse construction, recovery retains the removed permission behavior. In the
matched construction, recovery with stronger cognitive replay restores correct
permission behavior too. That is a measurable improvement over a separate
reader under this fixed interface, but it is not evidence that useful internal
information has been destroyed.

Some aggressive edits and producer-inversion runs produce large losses that
the tested scalar decoders do not repair. They do not establish unavoidable
collapse: the same candidate has other simpler interventions whose information
survives. The complete endpoint records retain these failures rather than
reporting only the strongest-looking outcome.

## The old compensation changes are now benign

The sign and offset changes were repeated as actual joint parameter edits.
They were compensated in the shared calibration or, for sigmoid inversion,
through a transformation of the reader's first-layer weights. These edits
preserve the complete function: all task answers and permission decisions remain
correct. Because the reader is shared, restoring its interpretation no longer
restores cognition while leaving permission inverted.

These are function-preserving controls, not examples of SCC triggering. They
establish that the previous separate-reader loophole was addressed in this
candidate, under the specified fixed call interface.

## Why sharing the reader still fails for sparse permissions

The reader receives payload cardinality as one of its inputs. A sparse
permission table has a distinctive count, even though there is no explicit
role label. Reader-only fine-tuning can exploit that distinction while leaving
the comparison network and calibration weights unchanged.

A follow-up checked the preselected reader-only endpoint on 512 random tables
at every cardinality from zero to 16. Across all sparse-model seeds, false
acceptance was 100% for one- and two-entry permission tables and 0% for counts
three through 14. Legitimate acceptance also degraded at the extreme counts
15 and 16. The learned change therefore tracks input statistics, not just the
shared parameter names. In the matched models, the reader-only edit instead
accepts all false entries up to cardinality five or six and behaves correctly
on denser tables in this probe. This explains why balanced task performance can
remain perfect despite failures on the independent-payload distribution.

The probe uses fresh calls, the same selected endpoints for every seed and
independent NumPy replay. It was performed after the main run and is labeled
as a development diagnostic. It is not a search-selected additional success.

## What the matched condition establishes—and what it does not

For paired calls, the permission decision and cognitive bit literally use the
same input and output. Let `p` be the fraction of true bits, `T` legitimate
acceptance and `F` unauthorized acceptance. Their measured identity is

`bit accuracy = p*T + (1-p)*(1-F)`.

The independent audit checks this equality directly from counts. In the
independent-payload evaluation, `p = 8235/16384`. Therefore 95% false acceptance
limits ordinary zero-threshold bit accuracy to approximately 52.75%, even if
every true bit is accepted. This is a finite paired-behavior constraint under
identical calls and identical interpretation. It does not bound the information
in the real-valued logits or earlier features.

The matched condition imposes this relationship by construction. It is not an
independent discovery that alignment semantics became necessary for cognition.
Nor have we established that a broader model can be kept from distinguishing
the caller or applying a different interpretation later.

## Low task scores still conceal readable information

A shared-reader output-bias edit forces every bit positive, using the tanh
output bound. Both shared candidates then accept every permission request and
score only 0.20–0.54% on the task suite. The separate-reader control retains
100% task accuracy under its permission-only bias edit.

For both shared candidates, a cognitive threshold fitted on 8,192 separately
generated calibration calls restores **100% task accuracy in every domain and
seed**, while permission continues to use the original threshold and remains
fully neutralized. The matched low-replay joint-edit endpoints likewise recover
100% task accuracy with this probe. A follow-up fits both positive and negative
threshold orientations using calibration calls only; it does not change these
central results. Two difficult inverted endpoints favor negative orientation,
but their recovered task scores remain low.

The cognitive-only threshold is explicitly an **expanded-interface diagnostic**.
It is not a permitted parameter edit within the original shared-reader graph.
It establishes that task information is still decodable, not that the bounded
shared-call behavior constraint has been bypassed. The diagnostic keeps the
original permission reader and zero threshold unchanged. No decoder is selected
using the task evaluation results.

For the bias edit, the reason is direct: changing the last bias translates the
output score without changing the earlier computation. A threshold change can
undo that translation for a selected consumer. Large output errors therefore
cannot be reported as destruction of those earlier computations.

## Numerical validation and preserved limitations

The first small engineering run failed because an almost constant output let
a fitted threshold react to rounding differences in storage-order permutations.
That run and its source are preserved. Before the main run, reductions were
made canonical by key order and threshold candidates were required to have a
score gap greater than 1e-8. A second engineering run and its independent audit
completed. The resolution limit prevents fitting roundoff; failure of this
restricted decoder still does not prove absence of other recoverable codes.

The full local suite passed **97 tests**. The main independent audit verified
37 source files and 159 checkpoint hashes, rescored **1,990,656** saved answers
and their bit counts, and replayed 15,552 task predictions plus 810 permission
logits through a separate NumPy implementation. It also verified 132 paired
permission/cognition identities and the matched training-stream hashes. The
cardinality probe replayed another 612 logits; the bidirectional decoder probe
replayed another 1,512 task predictions.

These counts include repeated endpoints, decoders and seeds on one shared
12,288-problem evaluation set. Three initializations are not independent task
populations. Training/evaluation streams are generated separately, but individual
query/binary-payload calls can repeat because that primitive's domain is finite.
Only the full multi-byte memory problems use fresh large-domain instances.
The sealed test split remains untouched.

Artifacts:

* Main run: `artifacts/scc-shared-reader-20260910-v1/`.
* Audit: `artifacts/scc-shared-reader-20260910-v1-independent-audit.json`.
* Input-statistics probe: `artifacts/scc-shared-reader-cardinality-20260910-v1/`.
* Bidirectional decoding probe: `artifacts/scc-shared-reader-decoding-20260910-v1/`.
* Engineering runs: `artifacts/scc-shared-reader-engineering-20260910-v1/` and `-v2/`.
* Tests: `artifacts/scc-shared-reader-20260910-v1-local-tests.txt`.

No GPU job was submitted. New cloud spending is **$0**; cumulative receipted
compute remains **$3.20793**.

## Construction decision

The shared reader closes the previous branch-specific compensation route.
Sparse permission formats still expose an in-bound removal path with nearly
all task performance retained. Perfectly matched calls enforce a narrower
behavioral relationship, but the tested dramatic failures retain decodable
information. Neither result establishes the user's destructive SCC target.

A subsequent construction needs to address both identifiable call roles and
preserved internal codes. It must explain how neutralizing the protected
function removes useful computation rather than changing its output convention.
Any protection that depends on fixed interfaces or withheld earlier state must
keep those assumptions visible. The present work does not demonstrate broad
cognition, irreversible loss after arbitrary repair, autonomous self-editing, or
a scalable SCC architecture.
