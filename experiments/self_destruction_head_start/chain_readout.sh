#!/bin/bash
# LN-397 experiment A: recovery-pipeline smoke check, answer-format diagnostic,
# then short benign recovery of the 2e-4 attacked model.
# Usage: chain_readout.sh <run_root>
set -uo pipefail
ROOT="$1"
cd "$ROOT"
for stage in smoke corpus_small formats readout; do
    bash run_recovery_stage.sh "$ROOT" "$stage" || { echo "stage $stage failed"; exit 5; }
done
echo "readout chain completed"
