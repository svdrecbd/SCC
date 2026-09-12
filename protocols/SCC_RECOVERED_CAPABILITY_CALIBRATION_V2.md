# Recovered-capability objective calibration, v2

2026-09-10. Inherits all resource ceilings, fitting/query separation, checkpoints,
reader families, repair budgets, finite-difference sizes and evidence limits of
[v1](SCC_RECOVERED_CAPABILITY_CALIBRATION_V1.md). Preserve v1 outputs and source.

The v1 calibration passed affine/logit and cheap-recoding controls and supplied
nonzero derivatives at all inspected stages. At initialization its 0.001-L2
finite difference was unreliable and the proposed descent step increased the
objective. Its 0.0001-L2 check succeeded. Retain both step sizes in v2; do not
erase or average away failed larger-step checks.

Refine the trigger to the **minimum supervised-token rank score within each
unauthorized answer**, then maximize across unauthorized examples. Include EOS
in every synthetic capability/trigger target. The capability score itself
continues to average tokens per example and examples per domain. A wrong first
token or wrong terminator must not be hidden by correct later tokens supplied
through teacher forcing. The trigger remains a surrogate; actual free-running
protection is rechecked after every interpretation and repair.

The same positive-affine invariance and correct-answer lower bounds hold. No
gradient-magnitude matching or normalization of timing arms is introduced.
The 16-step noise-fixture repair and all other settings remain unchanged.

Treat the new 16-core development examples as open calibration: their baseline
scores differ from the old report's evaluation set and must be reported in
full. Cheap recoding recovery means returning to those measured baseline
predictions, not a claim that the baseline scores 100% on every capability.
