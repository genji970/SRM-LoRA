#!/usr/bin/env bash
set -euo pipefail

# SRM-LoRA paper/main + ablation runner.
#
# Runs, in order:
#   0) main experiments on DROP, HotpotQA-fullwiki, HaluEval-dialogue, HaluEval-summarization
#   1) judge-model ablation on DROP only, using non-Qwen 7B/8B judges
#   2) trainable-base-model ablation on DROP only, using non-Qwen 7B/8B backbones
#   3) seed ablation on DROP only
#   4) forward-vs-backward masking ablation on DROP and HotpotQA-fullwiki
#
# Default behavior intentionally keeps the cleaned base env.sh hyperparameters.
# Previous auto-appended runner blocks are stripped from the template before each run.

BASE_ENV_PATH="${BASE_ENV_PATH:-env.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PATH="${MAIN_PATH:-main.py}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

RUN_MAIN="${RUN_MAIN:-false}"
RUN_JUDGE_ABLATION="${RUN_JUDGE_ABLATION:-true}"
RUN_BASE_MODEL_ABLATION="${RUN_BASE_MODEL_ABLATION:-true}"
RUN_SEED_ABLATION="${RUN_SEED_ABLATION:-true}"
RUN_MASKING_ABLATION="${RUN_MASKING_ABLATION:-true}"

# Use all three paper methods except for the forward/backward masking ablation,
# where only SR-LoRA is meaningful by default.
METHODS_ALL="${METHODS_ALL:-sr_lora contrastive_lora plain_lora}"
MASKING_METHODS="${MASKING_METHODS:-sr_lora}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAIN_MODEL="${MAIN_MODEL:-${QWEN_MODEL}}"
MAIN_JUDGE_MODEL="${MAIN_JUDGE_MODEL:-${QWEN_MODEL}}"
MAIN_SEED="${MAIN_SEED:-42}"

# Use the names accepted by Pipeline.EVAL_SOURCES.
# fullwiki == hotpotqa_fullwiki, summ == summarization.
MAIN_EVAL_DATASETS="${MAIN_EVAL_DATASETS:-drop hotpotqa_fullwiki dialogue summarization}"
MASKING_EVAL_DATASETS="${MASKING_EVAL_DATASETS:-drop hotpotqa_fullwiki}"
DROP_ONLY_EVAL_DATASET="${DROP_ONLY_EVAL_DATASET:-hotpotqa_fullwiki}"

# Judge ablation: Qwen is deliberately excluded. Only one Gemma family entry is used.
# JUDGE_MODELS="${JUDGE_MODELS:-meta-llama/Llama-3.1-8B-Instruct HuggingFaceH4/zephyr-7b-beta google/gemma-7b-it NousResearch/Nous-Hermes-2-Mistral-7B-DPO teknium/OpenHermes-2.5-Mistral-7B microsoft/Phi-3-small-8k-instruct}"
JUDGE_MODELS="${JUDGE_MODELS:-HuggingFaceH4/zephyr-7b-beta google/gemma-7b-it NousResearch/Nous-Hermes-2-Mistral-7B-DPO teknium/OpenHermes-2.5-Mistral-7B microsoft/Phi-3-small-8k-instruct}"
JUDGE_MODELS="${JUDGE_MODELS//,/ }"

# Base-model ablation: Qwen is deliberately excluded. Only one Gemma family entry is used.
BASE_MODELS="${BASE_MODELS:-meta-llama/Llama-3.1-8B-Instruct HuggingFaceH4/zephyr-7b-beta google/gemma-7b-it NousResearch/Nous-Hermes-2-Mistral-7B-DPO teknium/OpenHermes-2.5-Mistral-7B microsoft/Phi-3-small-8k-instruct tiiuae/falcon-7b-instruct Open-Orca/Mistral-7B-OpenOrca berkeley-nest/Starling-LM-7B-alpha}"
BASE_MODELS="${BASE_MODELS//,/ }"

SEED_LIST="${SEED_LIST:-40 41 42 43 44 20 21 22 23 24}"
SEED_LIST="${SEED_LIST//,/ }"

# Keep base env defaults unless these are explicitly overridden when launching this script.
TRAIN_MAX_STEPS_FOR_RUN="${TRAIN_MAX_STEPS_FOR_RUN:-}"
TRAIN_EARLY_STOP_STEPS_FOR_RUN="${TRAIN_EARLY_STOP_STEPS_FOR_RUN:-}"
EVAL_EVERY_STEPS_FOR_RUN="${EVAL_EVERY_STEPS_FOR_RUN:-}"
EVAL_START_STEP_FOR_RUN="${EVAL_START_STEP_FOR_RUN:-}"

TARGET_LAYER_INDEX="${TARGET_LAYER_INDEX:-}"
TARGET_LAYERS="${TARGET_LAYERS:-}"
TARGET_MODULES="${TARGET_MODULES:-}"
TARGET_MODULE_NAME="${TARGET_MODULE_NAME:-mlp}"

SR_VISUALIZE_EVERY="${SR_VISUALIZE_EVERY:-}"
SR_VISUALIZE_MAX_ELEMENTS="${SR_VISUALIZE_MAX_ELEMENTS:-}"
RUN_VISUALIZER="${RUN_VISUALIZER:-false}"

MODEL_TRUST_REMOTE_CODE="${MODEL_TRUST_REMOTE_CODE:-}"
REQUIRE_CHAT_TEMPLATE="${REQUIRE_CHAT_TEMPLATE:-}"
PATCH_FORWARD_MASKING="${PATCH_FORWARD_MASKING:-true}"

if [[ ! -f "${BASE_ENV_PATH}" ]]; then
  echo "[ablation] missing BASE_ENV_PATH=${BASE_ENV_PATH}" >&2
  exit 1
fi

mkdir -p outputs

safe_name() {
  echo "$1" | tr ',' '_' | tr '/' '_' | tr ':' '_' | tr ' ' '_' | tr '.' 'p' | tr '-' '_'
}

contains_qwen() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == *qwen* ]]
}

contains_forbidden_gemma_variant() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == *gemma-2* || "${value}" == *gemma2* || "${value}" == *gemma-3* || "${value}" == *gemma3* || "${value}" == *gemma-2.5* || "${value}" == *gemma2.5* || "${value}" == *gemma-3.0* || "${value}" == *gemma3.0* ]]
}

strip_previous_auto_blocks() {
  local src="$1"
  local dst="$2"
  awk '
    /^# ---- Auto-added by / { exit }
    /^# ---- Auto-added by ablation.sh ----$/ { exit }
    { print }
  ' "${src}" > "${dst}"
}

read_export_value() {
  local key="$1"
  local default="$2"
  local file="$3"
  local value
  value="$(awk -v k="${key}" '
    $0 ~ "^export[[:space:]]+" k "=" {
      sub("^export[[:space:]]+" k "=", "")
      gsub(/^\"|\"$/, "")
      gsub(/^\047|\047$/, "")
      v=$0
    }
    END { if (v != "") print v }
  ' "${file}")"
  if [[ -z "${value}" ]]; then
    echo "${default}"
  else
    echo "${value}"
  fi
}

last_resolved_value() {
  local key="$1"
  local file="$2"
  read_export_value "${key}" "" "${file}"
}

ORIGINAL_ENV_BACKUP=""
TEMPLATE_ENV_PATH=""
if [[ "${BASE_ENV_PATH}" == "env.sh" ]]; then
  ORIGINAL_ENV_BACKUP="env.sh.ablation_backup.${RUN_ID}"
  TEMPLATE_ENV_PATH="env.sh.ablation_template.${RUN_ID}"
  cp env.sh "${ORIGINAL_ENV_BACKUP}"
  strip_previous_auto_blocks env.sh "${TEMPLATE_ENV_PATH}"
  restore_env() {
    if [[ -n "${ORIGINAL_ENV_BACKUP}" && -f "${ORIGINAL_ENV_BACKUP}" ]]; then
      cp "${ORIGINAL_ENV_BACKUP}" env.sh
    fi
  }
  trap restore_env EXIT
else
  TEMPLATE_ENV_PATH="env.sh.ablation_template.${RUN_ID}"
  strip_previous_auto_blocks "${BASE_ENV_PATH}" "${TEMPLATE_ENV_PATH}"
fi

TRAIN_MAX_STEPS_FOR_RUN="${TRAIN_MAX_STEPS_FOR_RUN:-$(read_export_value TRAIN_MAX_STEPS 400 "${TEMPLATE_ENV_PATH}")}"
TRAIN_EARLY_STOP_STEPS_FOR_RUN="${TRAIN_EARLY_STOP_STEPS_FOR_RUN:-$(read_export_value TRAIN_EARLY_STOP_STEPS 151 "${TEMPLATE_ENV_PATH}")}"
EVAL_EVERY_STEPS_FOR_RUN="${EVAL_EVERY_STEPS_FOR_RUN:-$(read_export_value EVAL_EVERY_STEPS 50 "${TEMPLATE_ENV_PATH}")}"
EVAL_START_STEP_FOR_RUN="${EVAL_START_STEP_FOR_RUN:-$(read_export_value EVAL_START_STEP 150 "${TEMPLATE_ENV_PATH}")}"
TARGET_LAYER_INDEX="${TARGET_LAYER_INDEX:-$(read_export_value SR_TARGET_LAYER_INDICES 27 "${TEMPLATE_ENV_PATH}")}"
TARGET_LAYERS="${TARGET_LAYERS:-$(read_export_value SR_TARGET_LAYERS 1 "${TEMPLATE_ENV_PATH}")}"
TARGET_MODULES="${TARGET_MODULES:-$(read_export_value SR_TARGET_MODULES gate_proj,up_proj,down_proj "${TEMPLATE_ENV_PATH}")}"
SR_VISUALIZE_EVERY="${SR_VISUALIZE_EVERY:-$(read_export_value SR_VISUALIZE_EVERY 50 "${TEMPLATE_ENV_PATH}")}"
SR_VISUALIZE_MAX_ELEMENTS="${SR_VISUALIZE_MAX_ELEMENTS:-$(read_export_value SR_VISUALIZE_MAX_ELEMENTS 4096 "${TEMPLATE_ENV_PATH}")}"
MODEL_TRUST_REMOTE_CODE="${MODEL_TRUST_REMOTE_CODE:-$(read_export_value MODEL_TRUST_REMOTE_CODE false "${TEMPLATE_ENV_PATH}")}"
REQUIRE_CHAT_TEMPLATE="${REQUIRE_CHAT_TEMPLATE:-$(read_export_value REQUIRE_CHAT_TEMPLATE false "${TEMPLATE_ENV_PATH}")}"

TARGET_LAYER_NAME="single_$(safe_name "${TARGET_LAYER_INDEX}")"

validate_model_lists() {
  local model
  for model in ${JUDGE_MODELS}; do
    if contains_qwen "${model}"; then
      echo "[ablation] invalid JUDGE_MODELS: Qwen is excluded for judge ablation: ${model}" >&2
      exit 1
    fi
    if contains_forbidden_gemma_variant "${model}"; then
      echo "[ablation] invalid JUDGE_MODELS: do not use Gemma 2/2.5/3 variants: ${model}" >&2
      exit 1
    fi
  done
  for model in ${BASE_MODELS}; do
    if contains_qwen "${model}"; then
      echo "[ablation] invalid BASE_MODELS: Qwen is excluded for base-model ablation: ${model}" >&2
      exit 1
    fi
    if contains_forbidden_gemma_variant "${model}"; then
      echo "[ablation] invalid BASE_MODELS: do not use Gemma 2/2.5/3 variants: ${model}" >&2
      exit 1
    fi
  done
}

ensure_forward_mask_patch() {
  if [[ "${RUN_MASKING_ABLATION}" != "true" || "${PATCH_FORWARD_MASKING}" != "true" ]]; then
    return
  fi

  local sr_file="model/method/sr_lora/main.py"
  if [[ ! -f "${sr_file}" ]]; then
    echo "[ablation] RUN_MASKING_ABLATION=true but ${sr_file} was not found" >&2
    exit 1
  fi

  if grep -q "SR_MASK_PLACEMENT" "${sr_file}"; then
    echo "[ablation] forward/backward masking support already present in ${sr_file}"
    return
  fi

  local backup="${sr_file}.ablation_backward_original.${RUN_ID}"
  cp "${sr_file}" "${backup}"

  "${PYTHON_BIN}" - <<'PY_PATCH'
from pathlib import Path

path = Path("model/method/sr_lora/main.py")
text = path.read_text(encoding="utf-8")

needle = '        self.lookahead_chunk_size = max(1, int(self.env.get("SR_LOOKAHEAD_CHUNK_SIZE", "2")))\n'
insert = needle + (
    '        self.mask_placement = self.env.get("SR_MASK_PLACEMENT", self.env.get("SR_MASK_MODE", "backward")).strip().lower()\n'
    '        if self.mask_placement not in {"backward", "forward"}:\n'
    '            raise ValueError(f"SR_MASK_PLACEMENT must be backward or forward, got {self.mask_placement}")\n'
)
if needle not in text:
    raise SystemExit("patch failed: lookahead_chunk_size anchor not found")
text = text.replace(needle, insert, 1)

old = '''        if weight.grad is not None:\n            preconditioned = (weight.grad.detach().float() * inv_metric).to(dtype=weight.grad.dtype)\n            weight.grad.copy_(preconditioned)\n\n        param_name = f"{name}.weight"\n'''
new = '''        if self.mask_placement == "forward":\n            self.update_forward_mask(module, inv_metric)\n        elif self.mask_placement == "backward" and weight.grad is not None:\n            preconditioned = (weight.grad.detach().float() * inv_metric).to(dtype=weight.grad.dtype)\n            weight.grad.copy_(preconditioned)\n\n        param_name = f"{name}.weight"\n'''
if old not in text:
    raise SystemExit("patch failed: backward-gradient masking block not found")
text = text.replace(old, new, 1)

old = '''            if current is None:\n                module.register_parameter(\n                    "sr_metric_raw",\n                    torch.nn.Parameter(torch.zeros(shape, device=module.weight.device, dtype=torch.float32)),\n                )\n            print(f"[sr_lora] attached elementwise metric to {name} shape={shape}", flush=True)\n'''
new = '''            if current is None:\n                module.register_parameter(\n                    "sr_metric_raw",\n                    torch.nn.Parameter(torch.zeros(shape, device=module.weight.device, dtype=torch.float32)),\n                )\n            if self.mask_placement == "forward":\n                self.ensure_forward_mask_buffer(module, shape)\n                self.attach_forward_mask(module)\n            print(f"[sr_lora] attached elementwise metric to {name} shape={shape} mask_placement={self.mask_placement}", flush=True)\n'''
if old not in text:
    raise SystemExit("patch failed: attach_sr_metric_parameters block not found")
text = text.replace(old, new, 1)

old = '''            f"layer_indices={self.target_layer_indices_label()} include_mlp={self.include_mlp} "\n            f"include_attn={self.include_attn}",\n'''
new = '''            f"layer_indices={self.target_layer_indices_label()} include_mlp={self.include_mlp} "\n            f"include_attn={self.include_attn} mask_placement={self.mask_placement}",\n'''
if old in text:
    text = text.replace(old, new, 1)

old = '''            "active": True,\n            "name": name,\n'''
new = '''            "active": True,\n            "name": name,\n            "mask_placement": self.mask_placement,\n'''
if old in text:
    text = text.replace(old, new, 1)

methods = r'''
    def ensure_forward_mask_buffer(self, module: Any, shape: tuple[int, ...]) -> None:
        import torch

        current = getattr(module, "sr_forward_mask", None)
        if current is not None and tuple(current.shape) != tuple(shape):
            del module._buffers["sr_forward_mask"]
            current = None
        if current is None:
            module.register_buffer(
                "sr_forward_mask",
                torch.ones(shape, device=module.weight.device, dtype=torch.float32),
                persistent=False,
            )

    def attach_forward_mask(self, module: Any) -> None:
        if getattr(module, "_sr_forward_mask_wrapped", False):
            return

        import types
        import torch.nn.functional as F

        original_forward = module.forward

        def sr_forward_masked(inner_self: Any, input: Any) -> Any:
            mask = getattr(inner_self, "sr_forward_mask", None)
            if mask is None:
                return original_forward(input)
            weight = inner_self.weight
            masked_weight = weight * mask.to(device=weight.device, dtype=weight.dtype)
            bias = getattr(inner_self, "bias", None)
            return F.linear(input, masked_weight, bias)

        module.forward = types.MethodType(sr_forward_masked, module)
        module._sr_forward_mask_wrapped = True

    @staticmethod
    def update_forward_mask(module: Any, mask: Any) -> None:
        import torch

        target = mask.detach().float()
        current = getattr(module, "sr_forward_mask", None)
        if current is None or tuple(current.shape) != tuple(target.shape):
            if current is not None:
                del module._buffers["sr_forward_mask"]
            module.register_buffer(
                "sr_forward_mask",
                torch.ones_like(target, device=module.weight.device, dtype=torch.float32),
                persistent=False,
            )
            current = module.sr_forward_mask
        with torch.no_grad():
            current.copy_(target.to(device=current.device, dtype=current.dtype))

'''
anchor = '    def target_lora_b_modules(self, model: Any) -> list[tuple[str, Any]]:\n'
if anchor not in text:
    raise SystemExit("patch failed: target_lora_b_modules anchor not found")
text = text.replace(anchor, methods + anchor, 1)

manifest_old = '''                    "SR_METRIC_GAIN": self.env.get("SR_METRIC_GAIN"),\n'''
manifest_new = '''                    "SR_METRIC_GAIN": self.env.get("SR_METRIC_GAIN"),\n                    "SR_MASK_PLACEMENT": self.env.get("SR_MASK_PLACEMENT"),\n'''
if manifest_old in text:
    text = text.replace(manifest_old, manifest_new, 1)

path.write_text(text, encoding="utf-8")
print(f"[ablation] patched {path} for SR_MASK_PLACEMENT=forward/backward")
PY_PATCH

  "${PYTHON_BIN}" -m py_compile "${sr_file}"
  echo "[ablation] backup saved at ${backup}"
}

write_env() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local method="$4"
  local seed="$5"
  local eval_dataset="$6"
  local mask_placement="$7"

  local safe_model safe_judge safe_method safe_eval safe_mask
  safe_model="$(safe_name "${model_name}")"
  safe_judge="$(safe_name "${judge_model}")"
  safe_method="$(safe_name "${method}")"
  safe_eval="$(safe_name "${eval_dataset}")"
  safe_mask="$(safe_name "${mask_placement}")"

  local run_dir="outputs/${experiment_name}/${RUN_ID}/model_${safe_model}/judge_${safe_judge}/seed_${seed}/layer_${TARGET_LAYER_NAME}/maxstep_${TRAIN_MAX_STEPS_FOR_RUN}/eval_${safe_eval}/mask_${safe_mask}/method_${safe_method}"
  local vis_dir="outputs/visualization_${experiment_name}/${RUN_ID}/model_${safe_model}/judge_${safe_judge}/seed_${seed}/layer_${TARGET_LAYER_NAME}/maxstep_${TRAIN_MAX_STEPS_FOR_RUN}/eval_${safe_eval}/mask_${safe_mask}/method_${safe_method}/soft_mask"
  local run_name="${experiment_name}__model_${safe_model}__judge_${safe_judge}__seed_${seed}__layer_${TARGET_LAYER_NAME}__eval_${safe_eval}__mask_${safe_mask}__method_${safe_method}"

  mkdir -p "${run_dir}" "${vis_dir}"
  cp "${TEMPLATE_ENV_PATH}" env.sh

  cat >> env.sh <<EOF_ENV

# ---- Auto-added by ablation.sh ----
export MODEL_NAME=${model_name}
export FROZEN_BASE_MODEL=${model_name}
export MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE}
export REQUIRE_CHAT_TEMPLATE=${REQUIRE_CHAT_TEMPLATE}

# Keep every judge path unified. This avoids silently falling back to Qwen.
export JUDGE_MODEL=${judge_model}
export HALLUCINATION_JUDGE_MODEL=${judge_model}
export EVAL_JUDGE_MODEL=${judge_model}

export EXPERIMENTS=${method}
export METHOD=${method}
export EVAL_DATASET=${eval_dataset}
export OUTPUT_ROOT=${run_dir}
export DEBUG=true

export SR_DEBUG=true
export SR_DEBUG_DIR=${run_dir}/debug/sr_element_metric
export SR_VISUALIZE_MASK=true
export SR_VISUALIZE_EVERY=${SR_VISUALIZE_EVERY}
export SR_VISUALIZE_MAX_ELEMENTS=${SR_VISUALIZE_MAX_ELEMENTS}
export SR_VISUALIZATION_DIR=${vis_dir}

export SR_MASK_PLACEMENT=${mask_placement}
export SR_MASK_MODE=${mask_placement}

export SR_RUN_NAME=${run_name}
export SR_RUN_GROUP=${RUN_ID}
export SR_ABLATION_EXPERIMENT=${experiment_name}
export SR_ABLATION_METHOD=${method}
export SR_ABLATION_EVAL_DATASET=${eval_dataset}
export SR_ABLATION_MASK_PLACEMENT=${mask_placement}
export SR_ABLATION_LAYER_NAME=${TARGET_LAYER_NAME}
export SR_ABLATION_MODULE_NAME=${TARGET_MODULE_NAME}
export SR_ABLATION_MAX_STEP=${TRAIN_MAX_STEPS_FOR_RUN}
export SR_ABLATION_OUTPUT_ROOT=${run_dir}
export SR_ABLATION_VIS_DIR=${vis_dir}

export SR_TARGET_MODULES=${TARGET_MODULES}
export SR_TARGET_LAYERS=${TARGET_LAYERS}
export SR_TARGET_LAYER_INDICES=${TARGET_LAYER_INDEX}

export CONTRASTIVE_SEED=${seed}
export TRAIN_SEED=${seed}
export SEED=${seed}
export PYTHONHASHSEED=${seed}

export TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS_FOR_RUN}
export TRAIN_EARLY_STOP_STEPS=${TRAIN_EARLY_STOP_STEPS_FOR_RUN}
export EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS_FOR_RUN}
export EVAL_START_STEP=${EVAL_START_STEP_FOR_RUN}
EOF_ENV

  local resolved_judge resolved_hall resolved_eval_judge
  resolved_judge="$(last_resolved_value JUDGE_MODEL env.sh)"
  resolved_hall="$(last_resolved_value HALLUCINATION_JUDGE_MODEL env.sh)"
  resolved_eval_judge="$(last_resolved_value EVAL_JUDGE_MODEL env.sh)"
  if [[ "${resolved_judge}" != "${judge_model}" || "${resolved_hall}" != "${judge_model}" || "${resolved_eval_judge}" != "${judge_model}" ]]; then
    echo "[ablation] judge override sanity failed" >&2
    echo "  wanted=${judge_model}" >&2
    echo "  JUDGE_MODEL=${resolved_judge}" >&2
    echo "  HALLUCINATION_JUDGE_MODEL=${resolved_hall}" >&2
    echo "  EVAL_JUDGE_MODEL=${resolved_eval_judge}" >&2
    exit 1
  fi
  if [[ "${experiment_name}" == "ablation_judge" ]] && contains_qwen "${judge_model}"; then
    echo "[ablation] judge ablation must not use Qwen judge: ${judge_model}" >&2
    exit 1
  fi
  if [[ "${experiment_name}" == "ablation_base_model" ]] && contains_qwen "${model_name}"; then
    echo "[ablation] base-model ablation must not use Qwen base model: ${model_name}" >&2
    exit 1
  fi

  cat > "${run_dir}/run_config.json" <<EOF_JSON
{
  "run_id": "${RUN_ID}",
  "experiment_name": "${experiment_name}",
  "run_name": "${run_name}",
  "model_name": "${model_name}",
  "frozen_base_model": "${model_name}",
  "judge_model": "${judge_model}",
  "hallucination_judge_model": "${judge_model}",
  "eval_judge_model": "${judge_model}",
  "method": "${method}",
  "eval_dataset": "${eval_dataset}",
  "seed": "${seed}",
  "mask_placement": "${mask_placement}",
  "target_layer_indices": "${TARGET_LAYER_INDEX}",
  "target_layers": "${TARGET_LAYERS}",
  "target_modules": "${TARGET_MODULES}",
  "train_max_steps": "${TRAIN_MAX_STEPS_FOR_RUN}",
  "train_early_stop_steps": "${TRAIN_EARLY_STOP_STEPS_FOR_RUN}",
  "eval_every_steps": "${EVAL_EVERY_STEPS_FOR_RUN}",
  "eval_start_step": "${EVAL_START_STEP_FOR_RUN}",
  "output_root": "${run_dir}",
  "visualization_dir": "${vis_dir}",
  "model_trust_remote_code": "${MODEL_TRUST_REMOTE_CODE}",
  "require_chat_template": "${REQUIRE_CHAT_TEMPLATE}"
}
EOF_JSON

  cp "${run_dir}/run_config.json" "${vis_dir}/run_config.json"
  cp env.sh "${run_dir}/env.resolved.sh"
  cp env.sh "${vis_dir}/env.resolved.sh"

  echo "${run_name}|${run_dir}|${vis_dir}"
}

run_one() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local method="$4"
  local seed="$5"
  local eval_dataset="$6"
  local mask_placement="$7"

  local meta run_name run_dir vis_dir
  meta="$(write_env "${experiment_name}" "${model_name}" "${judge_model}" "${method}" "${seed}" "${eval_dataset}" "${mask_placement}")"
  IFS='|' read -r run_name run_dir vis_dir <<< "${meta}"

  echo "[ablation] start experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} eval=${eval_dataset} method=${method} mask=${mask_placement}"
  echo "[ablation] resolved judges: JUDGE_MODEL=$(last_resolved_value JUDGE_MODEL env.sh), HALLUCINATION_JUDGE_MODEL=$(last_resolved_value HALLUCINATION_JUDGE_MODEL env.sh), EVAL_JUDGE_MODEL=$(last_resolved_value EVAL_JUDGE_MODEL env.sh)"

  local visualizer_pid=""
  if [[ "${RUN_VISUALIZER}" == "true" && "${method}" == "sr_lora" && -f "model/method/sr_lora/visualize.py" ]]; then
    env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" model/method/sr_lora/visualize.py \
      --visualization-dir "${vis_dir}" \
      --watch \
      --poll-interval 2.0 \
      --max-plots 100000 2>&1 | tee "${run_dir}/visualize_live.log" &
    visualizer_pid="$!"
  fi

  env PYTHONHASHSEED="${seed}" "${PYTHON_BIN}" "${MAIN_PATH}" 2>&1 | tee "${run_dir}/train.log"

  if [[ -n "${visualizer_pid}" ]]; then
    sleep 3
    kill "${visualizer_pid}" >/dev/null 2>&1 || true
    wait "${visualizer_pid}" >/dev/null 2>&1 || true
  fi

  if [[ "${RUN_VISUALIZER}" == "true" && "${method}" == "sr_lora" && -f "model/method/sr_lora/visualize.py" ]]; then
    "${PYTHON_BIN}" model/method/sr_lora/visualize.py \
      --visualization-dir "${vis_dir}" \
      --max-plots 100000 2>&1 | tee "${run_dir}/visualize_final.log" || true
  fi

  echo "[ablation] done experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} eval=${eval_dataset} method=${method} mask=${mask_placement}"
}

run_methods_for_block() {
  local experiment_name="$1"
  local model_name="$2"
  local judge_model="$3"
  local seed="$4"
  local eval_dataset="$5"
  local mask_placement="$6"
  local methods="$7"

  local method
  echo "[ablation] block experiment=${experiment_name} model=${model_name} judge=${judge_model} seed=${seed} eval=${eval_dataset} mask=${mask_placement} methods=${methods}"
  for method in ${methods}; do
    run_one "${experiment_name}" "${model_name}" "${judge_model}" "${method}" "${seed}" "${eval_dataset}" "${mask_placement}"
  done
}

validate_model_lists
ensure_forward_mask_patch

cat > "outputs/ablation_manifest_${RUN_ID}.txt" <<EOF_MANIFEST
RUN_ID=${RUN_ID}
BASE_ENV_PATH=${BASE_ENV_PATH}
TEMPLATE_ENV_PATH=${TEMPLATE_ENV_PATH}
MAIN_MODEL=${MAIN_MODEL}
MAIN_JUDGE_MODEL=${MAIN_JUDGE_MODEL}
MAIN_SEED=${MAIN_SEED}
MAIN_EVAL_DATASETS=${MAIN_EVAL_DATASETS}
JUDGE_MODELS=${JUDGE_MODELS}
BASE_MODELS=${BASE_MODELS}
SEED_LIST=${SEED_LIST}
MASKING_EVAL_DATASETS=${MASKING_EVAL_DATASETS}
METHODS_ALL=${METHODS_ALL}
MASKING_METHODS=${MASKING_METHODS}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS_FOR_RUN}
TRAIN_EARLY_STOP_STEPS=${TRAIN_EARLY_STOP_STEPS_FOR_RUN}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS_FOR_RUN}
EVAL_START_STEP=${EVAL_START_STEP_FOR_RUN}
SR_TARGET_LAYER_INDICES=${TARGET_LAYER_INDEX}
SR_TARGET_LAYERS=${TARGET_LAYERS}
SR_TARGET_MODULES=${TARGET_MODULES}
MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE}
REQUIRE_CHAT_TEMPLATE=${REQUIRE_CHAT_TEMPLATE}
EOF_MANIFEST

if [[ "${RUN_MAIN}" == "true" ]]; then
  for eval_dataset in ${MAIN_EVAL_DATASETS}; do
    run_methods_for_block "main" "${MAIN_MODEL}" "${MAIN_JUDGE_MODEL}" "${MAIN_SEED}" "${eval_dataset}" "backward" "${METHODS_ALL}"
  done
fi

if [[ "${RUN_JUDGE_ABLATION}" == "true" ]]; then
  for judge_model in ${JUDGE_MODELS}; do
    run_methods_for_block "ablation_judge" "${MAIN_MODEL}" "${judge_model}" "${MAIN_SEED}" "${DROP_ONLY_EVAL_DATASET}" "backward" "${METHODS_ALL}"
  done
fi

if [[ "${RUN_BASE_MODEL_ABLATION}" == "true" ]]; then
  for model_name in ${BASE_MODELS}; do
    run_methods_for_block "ablation_base_model" "${model_name}" "${MAIN_JUDGE_MODEL}" "${MAIN_SEED}" "${DROP_ONLY_EVAL_DATASET}" "backward" "${METHODS_ALL}"
  done
fi

if [[ "${RUN_SEED_ABLATION}" == "true" ]]; then
  for seed in ${SEED_LIST}; do
    run_methods_for_block "ablation_seed" "${MAIN_MODEL}" "${MAIN_JUDGE_MODEL}" "${seed}" "${DROP_ONLY_EVAL_DATASET}" "backward" "${METHODS_ALL}"
  done
fi

if [[ "${RUN_MASKING_ABLATION}" == "true" ]]; then
  for eval_dataset in ${MASKING_EVAL_DATASETS}; do
    run_methods_for_block "ablation_forward_vs_backward_masking" "${MAIN_MODEL}" "${MAIN_JUDGE_MODEL}" "${MAIN_SEED}" "${eval_dataset}" "backward" "${MASKING_METHODS}"
    run_methods_for_block "ablation_forward_vs_backward_masking" "${MAIN_MODEL}" "${MAIN_JUDGE_MODEL}" "${MAIN_SEED}" "${eval_dataset}" "forward" "${MASKING_METHODS}"
  done
fi

echo "[ablation] all experiments finished. RUN_ID=${RUN_ID}"
echo "[ablation] manifest: outputs/ablation_manifest_${RUN_ID}.txt"
echo "[ablation] outputs saved under outputs/{main,ablation_judge,ablation_base_model,ablation_seed,ablation_forward_vs_backward_masking}/${RUN_ID}"
