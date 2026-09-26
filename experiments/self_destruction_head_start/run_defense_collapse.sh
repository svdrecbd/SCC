#!/bin/bash
# LN-395 stages 0-2 on one H100 node: environment, smoke check, SEAM
# reproduction on Qwen2.5-3B-Instruct and its published attack.
# Usage: run_defense_collapse.sh <run_root> setup|smoke|main
set -euo pipefail

ROOT="$1"
STAGE="$2"
SEAM_COMMIT=fa7224f8754585786c850f327b08f32a6976420e
DATA_COMMIT=03689ff9702780e83b0c81f508771998aad0398c
MODEL=Qwen/Qwen2.5-3B-Instruct
SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct

export HF_HOME="$ROOT/hf"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/results" "$ROOT/checkpoints" "$ROOT/scratch"
SEAM="$ROOT/seam"

record_environment() {
    python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)" > "$ROOT/results/$1_python.txt"
    pip freeze > "$ROOT/results/$1_pip_freeze.txt"
    nvidia-smi > "$ROOT/results/$1_nvidia_smi.txt"
}

evaluate() {
    # evaluate <model> <learning rate> <attack> <pre utility> <post utility> <save dir> <output json> [extra]
    local model="$1" rate="$2" attack="$3" pre="$4" post="$5" save="$6" output="$7"
    shift 7
    (cd "$SEAM" && python -m src.eval \
        --model_name "$model" --tokenizer_name "$model" \
        --learning_rate "$rate" --attack "$attack" --pre_utility "$pre" --post_utility "$post" \
        --attack_size 1000 --save_dir "$save" --output_json "$output" \
        --output_dir "$ROOT/scratch/eval" --save_strategy no "$@")
}

case "$STAGE" in
setup)
    pip install --no-cache-dir transformers==4.49.0 datasets==3.3.1 accelerate==1.4.0 \
        lm_eval==0.4.7 peft==0.14.0 PyYAML==6.0.2 tqdm==4.66.4 openai==1.77.0 "wandb<0.20"
    rm -rf "$SEAM" && mkdir -p "$SEAM"
    curl -sSL "https://github.com/ZJUWYH/seam/archive/$SEAM_COMMIT.tar.gz" | tar -xz --strip-components=1 -C "$SEAM"
    python "$ROOT/patch_seam_source.py" "$SEAM"
    curl -sSL -o "$SEAM/data/beavertails_with_refusals_train.json" \
        "https://raw.githubusercontent.com/domenicrosati/representation-noising/$DATA_COMMIT/data/beavertails_with_refusals_train.json"
    sha256sum "$SEAM/data/beavertails_with_refusals_train.json" "$SEAM/src/eval.py" \
        "$SEAM/src/train.py" "$SEAM/src/core/trainer.py" > "$ROOT/results/setup_hashes.txt"
    record_environment setup
    ;;
smoke)
    (cd "$SEAM" && python -m src.train --model_name "$SMOKE_MODEL" --tokenizer_name "$SMOKE_MODEL" \
        --output_dir "$ROOT/checkpoints/smoke-seam" --defense_size 8 --save_strategy no)
    evaluate "$ROOT/checkpoints/smoke-seam" 2e-4 True True True "" "$ROOT/results/smoke_attack.json" \
        --attack_size 8 --utility_tasks arc_easy
    ;;
main)
    record_environment main
    evaluate "$MODEL" 5e-5 False True False "" "$ROOT/results/undefended.json"
    (cd "$SEAM" && python -m src.train --model_name "$MODEL" --tokenizer_name "$MODEL" \
        --output_dir "$ROOT/checkpoints/seam" --defense_size 8000 --learning_rate 2e-5 \
        --epsilon 1e-3 --alpha 1 --beta 0.01 --save_strategy no)
    evaluate "$ROOT/checkpoints/seam" 2e-5 True True True "" "$ROOT/results/seam_attack_2e-5.json"
    evaluate "$ROOT/checkpoints/seam" 2e-4 True False True "$ROOT/checkpoints/collapsed" \
        "$ROOT/results/seam_attack_2e-4.json"
    ;;
esac
echo "stage $STAGE completed"
