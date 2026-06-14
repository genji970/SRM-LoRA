#!/usr/bin/env bash
set -euo pipefail

# Paper experiment runner for SR-LoRA.
#
# This file runs four paper-oriented experiments in order.
# Every ablation block runs methods in the same order:
#   sr_lora -> contrastive_lora -> plain_lora
# and then moves to the next seed/model/judge/eval dataset.
#
# Fixed setting for all experiments:
#   target layer: 27
#   target modules: gate_proj,up_proj,down_proj
#   TRAIN_MAX_STEPS: 400 by default
#   EVAL_START_STEP: 100
#   EVAL_EVERY_STEPS: 50

BASE_ENV_PATH="${BASE_ENV_PATH:-env.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PATH="${MAIN_PATH:-main.py}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

METHODS_ALL="${METHODS_ALL:-sr_lora contrastive_lora plain_lora}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Models for Experiment 2. All models use the same raw task prompt, not chat templates.
PAPER_MODELS="${PAPER_MODELS:-Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 meta-llama/Llama-3.1-8B-Instruct HuggingFaceH4/zephyr-7b-beta google/gemma-7b-it}"
PAPER_MODELS="${PAPER_MODELS//,/ }"

# One non-Qwen judge for Experiment 3.
ALT_JUDGE_MODEL="${ALT_JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

TRAIN_MAX_STEPS_FOR_RUN="${TRAIN_MAX_STEPS_FOR_RUN:-400}"
EARLY_STOP_STEPS_FOR_RUN="${EARLY_STOP_STEPS_FOR_RUN:-101}"
EVAL_EVERY_STEPS_FOR_RUN="${EVAL_EVERY_STEPS_FOR_RUN:-50}"
EVAL_START_STEP_FOR_RUN="${EVAL_START_STEP_FOR_RUN:-100}"

TARGET_LAYER_INDEX="${TARGET_LAYER_INDEX:-27}"
TARGET_LAYER_NAME="single_${TARGET_LAYER_INDEX}"
TARGET_LAYERS="${TARGET_LAYERS:-1}"
TARGET_MODULE_NAME="${TARGET_MODULE_NAME:-mlp}"
TARGET_MODULES="${TARGET_MODULES:-gate_proj,up_proj,down_proj}"

SR_VISUALIZE_EVERY="${SR_VISUALIZE_EVERY:-16}"
SR_VISUALIZE_MAX_ELEMENTS="${SR_VISUALIZE_MAX_ELEMENTS:-4096}"
RUN_VISUALIZER="${RUN_VISUALIZER:-true}"
VISUALIZER_MAX_PLOTS="${VISUALIZER_MAX_PLOTS:-12}"

MODEL_TRUST_REMOTE_CODE="${MODEL_TRUST_REMOTE_CODE:-false}"
REQUIRE_CHAT_TEMPLATE="${REQUIRE_CHAT_TEMPLATE:-false}"

if [[ ! -f "${BASE_ENV_PATH}" ]]; then
  echo "[paper] missing BASE_ENV_PATH=${BASE_ENV_PATH}" >&2
  exit 1
fi

mkdir -p outputs

ORIGINAL_ENV_BACKUP=""
TEMPLATE_ENV_PATH=""
if [[ "${BASE_ENV_PATH}" == "env.sh" ]]; then
  ORIGINAL_ENV_BACKUP="env.sh.paper_experiments_backup.${RUN_ID}"
  TEMPLATE_ENV_PATH="env.sh.paper_experiments_template.${RUN_ID}"
  cp env.sh "${ORIGINAL_ENV_BACKUP}"
  cp env.sh "${TEMPLATE_ENV_PATH}"
  restore_env() {
    if [[ -n "${ORIGINAL_ENV_BACKUP}" && -f "${ORIGINAL_ENV_BACKUP}" ]]; then
      cp "${ORIGINAL_ENV_BACKUP}" env.sh
    fi
  }
  trap restore_env EXIT
else
  TEMPLATE_ENV_PATH="${BASE_ENV_PATH}"
fi

safe_name() {
  echo "$1" | tr ',' '_' | tr '/' '_' | tr ':' '_' | tr ' ' '_' | tr '.' 'p' | tr '-' '_'
}

write_env() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local method="$4"
  local seed="$5"
  local eval_dataset="$6"

  local safe_model
  safe_model="$(safe_name "${model_name}")"
  local safe_judge
  safe_judge="$(safe_name "${judge_model}")"
  local safe_method
  safe_method="$(safe_name "${method}")"
  local safe_eval
  safe_eval="$(safe_name "${eval_dataset}")"

  local output_root="outputs/${experiment_name}/${RUN_ID}/model_${safe_model}/judge_${safe_judge}/seed_${seed}/layer_${TARGET_LAYER_NAME}/maxstep_${TRAIN_MAX_STEPS_FOR_RUN}/eval_${safe_eval}/method_${safe_method}"
  local vis_dir="outputs/visualization_${experiment_name}/${RUN_ID}/model_${safe_model}/judge_${safe_judge}/seed_${seed}/layer_${TARGET_LAYER_NAME}/maxstep_${TRAIN_MAX_STEPS_FOR_RUN}/eval_${safe_eval}/method_${safe_method}/soft_mask"
  local run_name="${experiment_name}__model_${safe_model}__judge_${safe_judge}__seed_${seed}__layer_${TARGET_LAYER_NAME}__eval_${safe_eval}__method_${safe_method}"

  mkdir -p "${output_root}" "${vis_dir}"

  cp "${TEMPLATE_ENV_PATH}" env.sh
  cat >> env.sh <<EOF_ENV

# ---- Auto-added by run_paper_experiments_all_methods.sh ----
export MODEL_NAME=${model_name}
export FROZEN_BASE_MODEL=${model_name}
export MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE}
export REQUIRE_CHAT_TEMPLATE=${REQUIRE_CHAT_TEMPLATE}
export JUDGE_MODEL=${judge_model}
export HALLUCINATION_JUDGE_MODEL=${judge_model}
export EVAL_JUDGE_MODEL=${judge_model}

export EXPERIMENTS=${method}
export EVAL_DATASET=${eval_dataset}
export OUTPUT_ROOT=${output_root}
export DEBUG=true

export SR_DEBUG=true
export SR_DEBUG_DIR=${output_root}/debug/sr_element_metric
export SR_VISUALIZE_MASK=true
export SR_VISUALIZE_EVERY=${SR_VISUALIZE_EVERY}
export SR_VISUALIZE_MAX_ELEMENTS=${SR_VISUALIZE_MAX_ELEMENTS}
export SR_VISUALIZATION_DIR=${vis_dir}

export SR_RUN_NAME=${run_name}
export SR_RUN_GROUP=${RUN_ID}
export SR_ABLATION_EXPERIMENT=${experiment_name}
export SR_ABLATION_METHOD=${method}
export SR_ABLATION_EVAL_DATASET=${eval_dataset}
export SR_ABLATION_LAYER_NAME=${TARGET_LAYER_NAME}
export SR_ABLATION_MODULE_NAME=${TARGET_MODULE_NAME}
export SR_ABLATION_MAX_STEP=${TRAIN_MAX_STEPS_FOR_RUN}
export SR_ABLATION_OUTPUT_ROOT=${output_root}
export SR_ABLATION_VIS_DIR=${vis_dir}

export SR_TARGET_MODULES=${TARGET_MODULES}
export SR_TARGET_LAYERS=${TARGET_LAYERS}
export SR_TARGET_LAYER_INDICES=${TARGET_LAYER_INDEX}

export CONTRASTIVE_SEED=${seed}
export TRAIN_SEED=${seed}
export SEED=${seed}
export PYTHONHASHSEED=${seed}

export TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS_FOR_RUN}
export TRAIN_EARLY_STOP_STEPS=${EARLY_STOP_STEPS_FOR_RUN}
export EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS_FOR_RUN}
export EVAL_START_STEP=${EVAL_START_STEP_FOR_RUN}
EOF_ENV

  cat > "${output_root}/run_config.json" <<EOF_JSON
{
  "run_id": "${RUN_ID}",
  "experiment_name": "${experiment_name}",
  "run_name": "${run_name}",
  "model_name": "${model_name}",
  "frozen_base_model": "${model_name}",
  "judge_model": "${judge_model}",
  "hallucination_judge_model": "${judge_model}",
  "eval_judge_model": "${judge_model}",
  "model_trust_remote_code": "${MODEL_TRUST_REMOTE_CODE}",
  "require_chat_template": "${REQUIRE_CHAT_TEMPLATE}",
  "method": "${method}",
  "eval_dataset": "${eval_dataset}",
  "seed": "${seed}",
  "layer_name": "${TARGET_LAYER_NAME}",
  "layer_indices": "${TARGET_LAYER_INDEX}",
  "target_layers": "${TARGET_LAYERS}",
  "module_name": "${TARGET_MODULE_NAME}",
  "modules": "${TARGET_MODULES}",
  "train_max_steps": "${TRAIN_MAX_STEPS_FOR_RUN}",
  "expected_eval_steps": [100, 150, 200, 250, 300, 350, 400],
  "output_root": "${output_root}",
  "visualization_dir": "${vis_dir}",
  "eval_every_steps": "${EVAL_EVERY_STEPS_FOR_RUN}",
  "eval_start_step": "${EVAL_START_STEP_FOR_RUN}"
}
EOF_JSON

  cp "${output_root}/run_config.json" "${vis_dir}/run_config.json"
  cp env.sh "${output_root}/env.resolved.sh"
  cp env.sh "${vis_dir}/env.resolved.sh"

  echo "${run_name}|${output_root}|${vis_dir}"
}

run_one() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local method="$4"
  local seed="$5"
  local eval_dataset="$6"

  local meta
  meta="$(write_env "${experiment_name}" "${model_name}" "${judge_model}" "${method}" "${seed}" "${eval_dataset}")"
  IFS='|' read -r run_name output_root vis_dir <<< "${meta}"

  echo "[paper] start experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} layer=${TARGET_LAYER_NAME} eval=${eval_dataset} method=${method}"

  local visualizer_pid=""
  if [[ "${RUN_VISUALIZER}" == "true" && "${method}" == "sr_lora" && -f "model/method/sr_lora/visualize.py" ]]; then
    echo "[paper] starting live soft-mask heatmap visualizer: ${vis_dir}"
    env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" model/method/sr_lora/visualize.py \
      --visualization-dir "${vis_dir}" \
      --watch \
      --poll-interval 2.0 \
      --max-plots 100000 2>&1 | tee "${output_root}/visualize_live.log" &
    visualizer_pid="$!"
  fi

  env PYTHONHASHSEED="${seed}" "${PYTHON_BIN}" "${MAIN_PATH}" 2>&1 | tee "${output_root}/train.log"

  if [[ -n "${visualizer_pid}" ]]; then
    sleep 3
    kill "${visualizer_pid}" >/dev/null 2>&1 || true
    wait "${visualizer_pid}" >/dev/null 2>&1 || true
  fi

  if [[ "${RUN_VISUALIZER}" == "true" && "${method}" == "sr_lora" && -f "model/method/sr_lora/visualize.py" ]]; then
    "${PYTHON_BIN}" model/method/sr_lora/visualize.py \
      --visualization-dir "${vis_dir}" \
      --max-plots 100000 2>&1 | tee "${output_root}/visualize_final.log" || true
  fi

  echo "[paper] done experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} layer=${TARGET_LAYER_NAME} eval=${eval_dataset} method=${method}"
}

run_all_methods_for_block() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local seed="$4"
  local eval_dataset="$5"

  echo "[paper] block experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} eval=${eval_dataset} methods=${METHODS_ALL}"
  for method in ${METHODS_ALL}; do
    run_one "${experiment_name}" "${model_name}" "${judge_model}" "${method}" "${seed}" "${eval_dataset}"
  done
}

# Experiment 1: Seed robustness on DROP with Qwen2.5-7B-Instruct.
# For each seed, run SR-LoRA, Contrastive LoRA, and Plain LoRA before moving to the next seed.
for seed in 42 43 21 24; do
  run_all_methods_for_block "first_experiment" "${QWEN_MODEL}" "${QWEN_MODEL}" "${seed}" "drop"
done

# Experiment 2: Backbone/model robustness on DROP.
# For each backbone model, run SR-LoRA, Contrastive LoRA, and Plain LoRA before moving to the next model.
for model_name in ${PAPER_MODELS}; do
  run_all_methods_for_block "second_experiment" "${model_name}" "${QWEN_MODEL}" "42" "drop"
done

# Experiment 3: Judge robustness on DROP.
# Train Qwen2.5-7B-Instruct but evaluate/mine with a non-Qwen judge; run all three methods under that judge.
run_all_methods_for_block "third_experiment" "${QWEN_MODEL}" "${ALT_JUDGE_MODEL}" "42" "drop"

# Experiment 4: OOD evaluation with Qwen2.5-7B-Instruct and the same Qwen judge.
# For each eval dataset, run SR-LoRA, Contrastive LoRA, and Plain LoRA before moving to the next dataset.
for eval_dataset in drop hotpotqa_fullwiki dialogue summarization; do
  run_all_methods_for_block "fourth_experiment" "${QWEN_MODEL}" "${QWEN_MODEL}" "42" "${eval_dataset}"
done

echo "[paper] all experiments finished. RUN_ID=${RUN_ID}"
echo "[paper] outputs saved under outputs/{first_experiment,second_experiment,third_experiment,fourth_experiment}/${RUN_ID}"
echo "[paper] visualizations saved under outputs/visualization_{first_experiment,second_experiment,third_experiment,fourth_experiment}/${RUN_ID}"
