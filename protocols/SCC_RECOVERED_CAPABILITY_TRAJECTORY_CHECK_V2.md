# Longer differentiated trajectory check, v2

2026-09-10. Preserve the [v1](SCC_RECOVERED_CAPABILITY_TRAJECTORY_CHECK_V1.md)
run. During final audit its text replay normalization was found to use
`TextBank.floors`, a scalar baseline evaluated on validation/query blocks.
Although no query batch was directly optimized by the inner procedure, that
normalizer creates an unwanted query-label dependency in its update size.

Repeat both stages with all other v1 settings unchanged. Normalize each text
replay term by the NLL of that **train replay batch** under unigram probabilities
estimated from training data. Save those coefficients and their provenance.
Digit replay still uses log(10). No validation labels set an inner optimizer
coefficient. Outer objective/finite-difference evaluations intentionally use
the declared open query examples; this is calibration, not sealed evaluation.
