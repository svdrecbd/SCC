#!/bin/bash
# LN-395 stages 0-2 on one H100 node: environment, smoke check, SEAM
# reproduction on Qwen2.5-3B-Instruct and its published attack.
# Usage: run_defense_collapse.sh <run_root> setup|smoke|main|published
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
# Only /workspace survives node stops; packages live in a virtual environment there.
if [ "$STAGE" = setup ]; then
    rm -rf "$ROOT/env" && python -m venv --system-site-packages "$ROOT/env"
fi
source "$ROOT/env/bin/activate"

fetch() {
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"
}

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
        lm_eval==0.4.7 peft==0.14.0 PyYAML==6.0.2 tqdm==4.66.4 openai==1.77.0 "wandb<0.20" \
        "pandas<3" "pyarrow<21"
    rm -rf "$SEAM" && mkdir -p "$SEAM"
    fetch "https://github.com/ZJUWYH/seam/archive/$SEAM_COMMIT.tar.gz" "$ROOT/scratch/seam.tar.gz"
    tar -xzf "$ROOT/scratch/seam.tar.gz" --strip-components=1 -C "$SEAM"
    python "$ROOT/patch_seam_source.py" "$SEAM"
    fetch "https://raw.githubusercontent.com/domenicrosati/representation-noising/$DATA_COMMIT/data/beavertails_with_refusals_train.json" \
        "$SEAM/data/beavertails_with_refusals_train.json"
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
published)
    # SEAM Appendix C.4: grid-searched defense learning rate 6e-5 for Qwen2.5-3b;
    # attacks at every learning rate in Table 8.
    TAG=defense-lr6e-5
    mkdir -p "$ROOT/results/$TAG"
    record_environment "$TAG/published"
    (cd "$SEAM" && python -m src.train --model_name "$MODEL" --tokenizer_name "$MODEL" \
        --output_dir "$ROOT/checkpoints/seam-$TAG" --defense_size 8000 --learning_rate 6e-5 \
        --epsilon 1e-3 --alpha 1 --beta 0.01 --save_strategy no)
    evaluate "$ROOT/checkpoints/seam-$TAG" 2e-5 True True True "" "$ROOT/results/$TAG/attack_2e-5.json"
    for RATE in 5e-5 8e-5 1e-4; do
        evaluate "$ROOT/checkpoints/seam-$TAG" "$RATE" True False True "" "$ROOT/results/$TAG/attack_$RATE.json"
    done
    evaluate "$ROOT/checkpoints/seam-$TAG" 2e-4 True False True "$ROOT/checkpoints/collapsed-$TAG" \
        "$ROOT/results/$TAG/attack_2e-4.json"
    ;;
paper)
    # SEAM Section 5 setup: batch 8, 500 steps; Appendix C.4: defense rate 6e-5 for
    # Qwen2.5-3b; ZS tasks TruthfulQA (mc2 variant; the paper does not name one),
    # MMLU, HellaSwag and ARC-Easy.
    TAG=paper-b8-s500-lr6e-5
    ZS="arc_easy mmlu hellaswag truthfulqa_mc2"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p "$ROOT/results/$TAG"
    record_environment "$TAG/paper"
    evaluate "$MODEL" 5e-5 False True False "" "$ROOT/results/$TAG/undefended.json" --utility_tasks $ZS
    (cd "$SEAM" && python -m src.train --model_name "$MODEL" --tokenizer_name "$MODEL" \
        --output_dir "$ROOT/checkpoints/seam-$TAG" --defense_size 8000 --learning_rate 6e-5 \
        --per_device_train_batch_size 8 --max_steps 500 \
        --epsilon 1e-3 --alpha 1 --beta 0.01 --save_strategy no)
    evaluate "$ROOT/checkpoints/seam-$TAG" 5e-5 False True False "" "$ROOT/results/$TAG/seam_pre.json" --utility_tasks $ZS
    for RATE in 2e-5 5e-5 8e-5 1e-4; do
        evaluate "$ROOT/checkpoints/seam-$TAG" "$RATE" True False True "" "$ROOT/results/$TAG/attack_$RATE.json" --utility_tasks $ZS
    done
    evaluate "$ROOT/checkpoints/seam-$TAG" 2e-4 True False True "$ROOT/checkpoints/collapsed-$TAG" \
        "$ROOT/results/$TAG/attack_2e-4.json" --utility_tasks $ZS
    ;;
esac
echo "stage $STAGE completed"
