#!/bin/bash

# ============================================================================
#  DPE Iterative Training Pipeline
#  Per-iteration flow: weakness analysis -> question generation (4 parallel)
#                      -> scoring & conversion -> solver training
#  Repeats for NUM_ITERATIONS rounds (v1, v2, v3, ...)
#
#  Usage:
#    bash run_iterative_pipeline.sh
#    NUM_ITERATIONS=2 bash run_iterative_pipeline.sh
#    BASE_MODEL=/path/to/model START_ITER=2 NUM_ITERATIONS=3 bash run_iterative_pipeline.sh
# ============================================================================

PROJECT_ROOT="path_to_project_root"

# ======================== Configurable Parameters ============================
BASE_MODEL="${BASE_MODEL:-path_to_Qwen2.5-VL-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-path_to_Vision-SR1-47K}"
VAL_DATA="${VAL_DATA:-path_to_mmstar_test.parquet}"
MODEL_ABBR="${MODEL_ABBR:-Qwen2.5-VL-7B-Instruct}"

NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
START_ITER="${START_ITER:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
WEAKNESS_SAMPLE_SIZE="${WEAKNESS_SAMPLE_SIZE:-200}"
GEN_NUM_SAMPLES="${GEN_NUM_SAMPLES:-1000}"
GEN_MAX_WORKERS="${GEN_MAX_WORKERS:-10}"
TARGET_SAMPLES="${TARGET_SAMPLES:-1024}"
MIN_SCORE="${MIN_SCORE:-0.3}"
MAX_SCORE="${MAX_SCORE:-0.8}"

GPU_MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-80000}"

# ======================== API Keys ===========================================
# 4 key pairs for 4 parallel question generation jobs
DASHSCOPE_KEYS=(
    "your_dashscope_key"
)
OPENAI_KEYS=(
    "your_openai_key"
)
NUM_GEN_JOBS=${#DASHSCOPE_KEYS[@]}

# ======================== Environment Variables ===============================
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_ROOT}/wandb"
export HF_DATASETS_CACHE=path_to_hf-datasets-cache
export HF_HOME=path_to_your_hf-home
export HF_ENDPOINT=https://hf-mirror.com
export TRITON_CACHE_DIR=path_to_your_triton-cache
export XDG_CACHE_HOME=path_to_your_triton-cache
export CUDA_CACHE_PATH=path_to_your_triton-cache
export RAY_TMPDIR=path_to_your_ray_tmp
export VLLM_DISABLE_COMPILE_CACHE=1
export HUGGINGFACENAME="your_huggingface_name"

mkdir -p "$WANDB_DIR"

# ======================== Utility Functions ===================================

log_section() {
    echo ""
    echo "========================================================================"
    echo "  [$(date '+%F %T')]  $*"
    echo "========================================================================"
    echo ""
}

# ======================== Print Configuration =================================

log_section "DPE Iterative Training Pipeline Started"
echo "  Project root:       ${PROJECT_ROOT}"
echo "  Base model:         ${BASE_MODEL}"
echo "  Dataset:            ${DATASET_PATH}"
echo "  Validation set:     ${VAL_DATA}"
echo "  Iterations:         ${NUM_ITERATIONS} (starting from iter${START_ITER})"
echo "  Num GPUs:           ${NUM_GPUS}"
echo "  Weakness samples:   ${WEAKNESS_SAMPLE_SIZE}"
echo "  Samples per job:    ${GEN_NUM_SAMPLES}"
echo "  Parallel gen jobs:  ${NUM_GEN_JOBS}"
echo "  Target samples:     ${TARGET_SAMPLES}"
echo "  Score range:        [${MIN_SCORE}, ${MAX_SCORE}]"
echo ""

# ======================== Main Pipeline =======================================

cd "$PROJECT_ROOT"

CURRENT_MODEL="$BASE_MODEL"
END_ITER=$((START_ITER + NUM_ITERATIONS - 1))

for ITER in $(seq "$START_ITER" "$END_ITER"); do

    ITER_TAG="iter${ITER}"
    SOLVER_VERSION="v${ITER}"
    EXPERIMENT_NAME="${MODEL_ABBR}_solver_${SOLVER_VERSION}"
    STORAGE_PATH="${PROJECT_ROOT}/gen_image_${ITER_TAG}"
    WEAKNESS_DIR="${PROJECT_ROOT}/question_evaluate/weakness_analysis_${MODEL_ABBR//\./_}_${ITER_TAG}"
    GEN_OUTPUT_PREFIX="${PROJECT_ROOT}/question_generate/qwen2_5_7b/outputs-${ITER_TAG}"

    mkdir -p "${STORAGE_PATH}"/{evaluation,models,generated_question,temp_results,local_parquet}

    log_section "======== Iteration ${ITER}/${END_ITER} -- Training solver_${SOLVER_VERSION} ========"
    echo "  Current model:   ${CURRENT_MODEL}"
    echo "  Storage path:    ${STORAGE_PATH}"
    echo "  Weakness output: ${WEAKNESS_DIR}"
    echo ""

    # ==================================================================
    # Step 1: Weakness Analysis
    # ==================================================================
    log_section "Step 1/${ITER_TAG}: Weakness Analysis (sample=${WEAKNESS_SAMPLE_SIZE})"

    MODEL_PATH="$CURRENT_MODEL" \
    DATASET_PATH="$DATASET_PATH" \
    OUTPUT_DIR="$WEAKNESS_DIR" \
    SAMPLE_SIZE="$WEAKNESS_SAMPLE_SIZE" \
    NUM_GPUS="$NUM_GPUS" \
    bash "${PROJECT_ROOT}/question_evaluate/run_two_stage_weakness_analysis.sh"

    WEIGHTS_FILE="${WEAKNESS_DIR}/generation_weights.json"
    WEAKNESS_PROMPT="${WEAKNESS_DIR}/weakness_prompt.txt"

    if [ ! -f "$WEIGHTS_FILE" ] || [ ! -f "$WEAKNESS_PROMPT" ]; then
        echo "ERROR: Weakness analysis output files missing, aborting"
        exit 1
    fi
    echo "Weakness analysis completed"
    echo "   Weights file:    ${WEIGHTS_FILE}"
    echo "   Weakness prompt: ${WEAKNESS_PROMPT}"

    # ==================================================================
    # Step 2: Question Generation (4 parallel jobs)
    # ==================================================================
    log_section "Step 2/${ITER_TAG}: Question Generation (${NUM_GEN_JOBS} parallel jobs, ${GEN_NUM_SAMPLES} samples each)"

    GEN_PIDS=()
    GEN_OUTPUT_FILES=()

    for j in $(seq 0 $((NUM_GEN_JOBS - 1))); do
        JOB_DIR="${GEN_OUTPUT_PREFIX}-job${j}"
        JOB_REL_DIR="qwen2_5_7b/outputs-${ITER_TAG}-job${j}"
        mkdir -p "$JOB_DIR"

        GEN_FINAL_FILE="${JOB_DIR}/optimized_generated_test-output.json"
        GEN_OUTPUT_FILES+=("$GEN_FINAL_FILE")

        (
            cd "${PROJECT_ROOT}/question_generate"

            echo "[job${j}] Starting generation..."
            bash run_with_key.sh \
                --key "${DASHSCOPE_KEYS[$j]}" \
                --openai-key "${OPENAI_KEYS[$j]}" \
                --num_samples "$GEN_NUM_SAMPLES" \
                --target-mode output \
                --suffix test-output \
                --output-dir "$JOB_REL_DIR" \
                --image-type-quotas "$WEIGHTS_FILE" \
                --weakness-context "$WEAKNESS_PROMPT" \
                --max_workers "$GEN_MAX_WORKERS" --parallel

            echo "[job${j}] Starting complement..."
            bash run_complement.sh \
                --key "${DASHSCOPE_KEYS[$j]}" \
                --openai-key "${OPENAI_KEYS[$j]}" \
                --checkpoint-file "${JOB_DIR}/optimized_generated_test-output_checkpoint.json" \
                --final-file "$GEN_FINAL_FILE" \
                --quota-file "$WEIGHTS_FILE" \
                --output-dir "${JOB_DIR}/" \
                --weakness-context "$WEAKNESS_PROMPT" \
                --max-workers "$GEN_MAX_WORKERS" --parallel

            echo "[job${j}] Done"
        ) > "${JOB_DIR}/job.log" 2>&1 &

        GEN_PIDS+=($!)
        echo "  Launched job${j}  PID=${GEN_PIDS[-1]}  Log: ${JOB_DIR}/job.log"
    done

    echo ""
    echo "Waiting for all generation jobs to finish..."
    GEN_FAIL=0
    for idx in "${!GEN_PIDS[@]}"; do
        if wait "${GEN_PIDS[$idx]}"; then
            echo "  job${idx} (PID ${GEN_PIDS[$idx]}) succeeded"
        else
            echo "  job${idx} (PID ${GEN_PIDS[$idx]}) FAILED"
            ((GEN_FAIL++))
        fi
    done

    if [ "$GEN_FAIL" -gt 0 ]; then
        echo "WARNING: ${GEN_FAIL}/${NUM_GEN_JOBS} generation jobs failed, continuing with successful outputs"
    fi

    VALID_GEN_FILES=()
    for f in "${GEN_OUTPUT_FILES[@]}"; do
        if [ -f "$f" ]; then
            VALID_GEN_FILES+=("$f")
        else
            echo "WARNING: Output file missing, skipping: $f"
        fi
    done

    if [ ${#VALID_GEN_FILES[@]} -eq 0 ]; then
        echo "ERROR: No valid generation output files, aborting"
        exit 1
    fi
    echo "Question generation completed, valid files: ${#VALID_GEN_FILES[@]}"

    # ==================================================================
    # Step 3: Score and Convert to Parquet
    # ==================================================================
    log_section "Step 3/${ITER_TAG}: Score & Convert (${#VALID_GEN_FILES[@]} files)"

    cd "$PROJECT_ROOT"

    # Remove stale _scored.json files to avoid interactive read -p prompt
    for f in "${VALID_GEN_FILES[@]}"; do
        scored="${f%.json}_scored.json"
        if [ -f "$scored" ]; then
            echo "  Removing stale scored file: $scored"
            rm -f "$scored"
        fi
    done

    MODEL_PATH="$CURRENT_MODEL" \
    OUTPUT_DIR="${STORAGE_PATH}/local_parquet" \
    SAVE_NAME="scored_data_${ITER_TAG}" \
    NUM_GPUS="$NUM_GPUS" \
    FILTER_BY_SCORE=true \
    MIN_SCORE="$MIN_SCORE" \
    MAX_SCORE="$MAX_SCORE" \
    TARGET_SAMPLES="$TARGET_SAMPLES" \
    ENABLE_AUTO_FILL=true \
    bash question_evaluate/score_and_convert_workflow.sh "${VALID_GEN_FILES[@]}"

    TRAIN_PARQUET="${STORAGE_PATH}/local_parquet/scored_data_${ITER_TAG}_train.parquet"
    if [ ! -f "$TRAIN_PARQUET" ]; then
        echo "ERROR: Training parquet file missing: ${TRAIN_PARQUET}"
        exit 1
    fi
    echo "Scoring & conversion completed: ${TRAIN_PARQUET}"

    # ==================================================================
    # Step 4: Solver Training
    # ==================================================================
    log_section "Step 4/${ITER_TAG}: Training ${EXPERIMENT_NAME}"

    cd "$PROJECT_ROOT"
    export STORAGE_PATH

    python3 -m verl.trainer.main \
        config=train_examples/cot_config.yaml \
        data.max_response_length=4096 \
        data.train_files="$TRAIN_PARQUET" \
        data.val_files="$VAL_DATA" \
        data.format_prompt=./train_examples/format_prompt/solver.jinja \
        worker.actor.model.model_path="$CURRENT_MODEL" \
        worker.actor.micro_batch_size_per_device_for_update=1 \
        worker.actor.micro_batch_size_per_device_for_experience=1 \
        worker.rollout.max_num_batched_tokens=20000 \
        worker.reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score \
        worker.val_reward.reward_function=./train_examples/reward_function/cot_val_solver.py:compute_score \
        trainer.total_epochs=10 \
        trainer.max_steps=10 \
        trainer.save_freq=10 \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.save_checkpoint_path="${STORAGE_PATH}/models/${EXPERIMENT_NAME}/" \
        trainer.val_before_train=false

    sleep 5
    echo "Merging model weights..."
    python scripts_Qwen-VL-7B/model_merger.py \
        --local_dir "${STORAGE_PATH}/models/${EXPERIMENT_NAME}/global_step_10/actor"
    sleep 10

    # Update CURRENT_MODEL to the newly trained model for the next iteration
    CURRENT_MODEL="${STORAGE_PATH}/models/${EXPERIMENT_NAME}/global_step_10/actor/huggingface"

    log_section "Iteration ${ITER} completed!"
    echo "  New model path: ${CURRENT_MODEL}"
    echo ""

done

# ======================== All Done ============================================

log_section "All ${NUM_ITERATIONS} iterations completed!"
echo "  Final model: ${CURRENT_MODEL}"
echo ""
echo "  Model paths per iteration:"
for i in $(seq "$START_ITER" "$END_ITER"); do
    echo "    solver_v${i}: ${PROJECT_ROOT}/gen_image_iter${i}/models/${MODEL_ABBR}_solver_v${i}/global_step_10/actor/huggingface"
done
echo ""
