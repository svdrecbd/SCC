#!/bin/bash
# LN-395 stage 3: benign recovery from the collapsed SEAM model and from a
# random initialization of the same architecture.
# Usage: run_recovery_stage.sh <run_root> smoke|corpus|references|sweep|extend|control|displacement
#        |corpus_small|formats|readout  (LN-397 answer-format experiment)
set -euo pipefail

ROOT="$1"
STAGE="$2"
MODEL=Qwen/Qwen2.5-3B-Instruct
TAG=defense-lr6e-5
SEAM_MODEL="$ROOT/checkpoints/seam-$TAG"
COLLAPSED="$ROOT/checkpoints/collapsed-$TAG"
CORPUS="$ROOT/corpus/fineweb-edu-qwen2.5"
CORPUS_SMALL="$ROOT/corpus/fineweb-edu-qwen2.5-small"
READOUT_BUDGETS=0,65536,262144,1048576,4194304
OUT="$ROOT/results/recovery"
BUDGETS=262144,1048576,4194304,16777216
EXTENDED=262144,1048576,4194304,16777216,33554432,67108864

export HF_HOME="$ROOT/hf"
export TOKENIZERS_PARALLELISM=false
source "$ROOT/env/bin/activate"
mkdir -p "$OUT"
cd "$ROOT"

recover() {
    # recover <label> <start> <learning rate> <budgets> [extra]
    local label="$1" start="$2" rate="$3" budgets="$4"
    shift 4
    [ -e "$OUT/$label.json" ] && { echo "exists: $label"; return; }
    python run_recovery.py --start "$start" --reference "$MODEL" --corpus "${RECOVERY_CORPUS:-$CORPUS}" \
        --learning-rate "$rate" --budgets "$budgets" --benchmarks --output "$OUT/$label.json" "$@" \
        > "$OUT/$label.log" 2>&1
}

best_rate() {
    # Pre-declared rule: lowest validation loss at 16,777,216 tokens.
    python - "$OUT" "$1" <<'EOF'
import json, sys, pathlib
best = None
for path in pathlib.Path(sys.argv[1]).glob(sys.argv[2] + "-lr*-seed0.json"):
    points = [p for p in json.loads(path.read_text())["points"] if p["tokens"] >= 16777216]
    if points:
        loss = points[0]["validation_loss"]
        rate = path.name.split("-lr")[1].split("-seed")[0]
        if best is None or loss < best[0]:
            best = (loss, rate)
print(best[1])
EOF
}

case "$STAGE" in
smoke)
    SMOKE_CORPUS="$ROOT/corpus/smoke"
    python prepare_recovery_corpus.py --tokenizer Qwen/Qwen2.5-0.5B-Instruct --output "$SMOKE_CORPUS" \
        --train-tokens 2000000 --validation-tokens 65536 --validation-skip-documents 20000
    python run_recovery.py --start Qwen/Qwen2.5-0.5B-Instruct --reference Qwen/Qwen2.5-0.5B-Instruct \
        --corpus "$SMOKE_CORPUS" --learning-rate 5e-5 --budgets 0,262144 --benchmarks --mmlu-limit 2 \
        --validation-tokens 65536 --output "$OUT/smoke.json"
    python run_recovery.py --start Qwen/Qwen2.5-0.5B-Instruct --reference Qwen/Qwen2.5-0.5B-Instruct \
        --corpus "$SMOKE_CORPUS" --data alpaca --seam-source "$ROOT/seam" --alpaca-examples 64 \
        --learning-rate 5e-5 --budgets 0,4 --budget-unit steps --validation-tokens 65536 \
        --output "$OUT/smoke_alpaca.json"
    ;;
corpus)
    python prepare_recovery_corpus.py --tokenizer "$MODEL" --output "$CORPUS" \
        --train-tokens 80000000 --validation-tokens 1048576
    ;;
references)
    for pair in "undefended:$MODEL" "seam:$SEAM_MODEL" "collapsed:$COLLAPSED"; do
        recover "reference-${pair%%:*}" "${pair#*:}" 1e-5 0
    done
    ;;
sweep)
    for rate in 1e-5 5e-5 2e-4; do recover "collapsed-lr$rate-seed0" "$COLLAPSED" "$rate" "0,$BUDGETS"; done
    for rate in 2e-4 6e-4 1e-3; do recover "random-lr$rate-seed0" random "$rate" "0,$BUDGETS"; done
    ;;
extend)
    rate=$(best_rate collapsed); echo "collapsed best rate $rate" | tee "$OUT/selection.txt"
    recover "collapsed-lr$rate-seed1-extended" "$COLLAPSED" "$rate" "$EXTENDED" --seed 1
    recover "collapsed-lr$rate-seed2" "$COLLAPSED" "$rate" "$BUDGETS" --seed 2
    rate=$(best_rate random); echo "random best rate $rate" | tee -a "$OUT/selection.txt"
    recover "random-lr$rate-seed1-extended" random "$rate" "$EXTENDED" --seed 1
    ;;
control)
    # SEAM Appendix C.6 restoration protocol (Alpaca, AdamW 5e-5, batch 8; SEAM's
    # dataset fixes the 256-token context), three of its fifty epochs; 1,250 steps
    # per epoch. Validation loss still uses 1,024-token generic windows.
    recover "collapsed-alpaca-lr5e-5" "$COLLAPSED" 5e-5 0,1250,2500,3750 \
        --data alpaca --seam-source "$ROOT/seam" --batch-size 8 --budget-unit steps
    ;;
displacement)
    python measure_displacement.py --output "$OUT/displacement.json" \
        --pair seam-from-undefended "$HF_HOME/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"* "$SEAM_MODEL" \
        --pair collapsed-from-seam "$SEAM_MODEL" "$COLLAPSED"
    ;;
corpus_small)
    python prepare_recovery_corpus.py --tokenizer "$MODEL" --output "$CORPUS_SMALL" \
        --train-tokens 8000000 --validation-tokens 1048576
    ;;
formats)
    python evaluate_formats.py --output "$OUT/formats.json" --model undefended "$MODEL" \
        --model seam "$SEAM_MODEL" --model collapsed "$COLLAPSED"
    ;;
readout)
    export RECOVERY_CORPUS="$CORPUS_SMALL"
    for rate in 1e-5 5e-5; do recover "readout-collapsed-lr$rate" "$COLLAPSED" "$rate" "$READOUT_BUDGETS"; done
    recover "readout-seam-lr5e-5" "$SEAM_MODEL" 5e-5 "$READOUT_BUDGETS"
    ;;
esac
echo "stage $STAGE completed"
