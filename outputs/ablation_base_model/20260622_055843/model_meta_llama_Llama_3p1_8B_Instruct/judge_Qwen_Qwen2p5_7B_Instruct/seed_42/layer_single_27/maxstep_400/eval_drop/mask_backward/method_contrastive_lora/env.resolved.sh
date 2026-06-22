export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export MODEL_DTYPE=bfloat16
export MODEL_DEVICE_MAP=auto
export MODEL_LOW_CPU_MEM_USAGE=true

export FROZEN_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
export HALLUCINATION_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
export EVAL_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct

export FROZEN_BASE_MAX_PROMPT_LENGTH=16384
export FROZEN_BASE_MAX_NEW_TOKENS=16

export JUDGE_MAX_PROMPT_LENGTH=16384
export JUDGE_MAX_NEW_TOKENS=16
export JUDGE_BATCH_SIZE=32

export HALLUCINATION_JUDGE_MAX_PROMPT_LENGTH=16384
export HALLUCINATION_JUDGE_MAX_NEW_TOKENS=16
export HALLUCINATION_JUDGE_BATCH_SIZE=32

export EVAL_JUDGE_MAX_PROMPT_LENGTH=16384
export EVAL_JUDGE_MAX_NEW_TOKENS=16
export EVAL_JUDGE_BATCH_SIZE=32

export EXPERIMENTS=sr_lora,contrastive_lora,plain_lora

export LORA_R=16
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export LORA_BIAS=none
export LORA_TASK_TYPE=CAUSAL_LM
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

# SR-LoRA closed-form metric settings
export SR_INCLUDE_MLP=true
export SR_INCLUDE_ATTN=false
export SR_TARGET_MODULES=gate_proj,up_proj,down_proj

# Use only layer index 27.
export SR_TARGET_LAYER_INDICES=27
export SR_TARGET_LAYERS=1

# Core SR metric hyperparameters
export SR_FINITE_DIFF_EPS=1e-8
export SR_X_CLIP=10.0
export SR_METRIC_GAIN=8.0
export SR_METRIC_MIN=1.0
export SR_METRIC_MAX=8.0

export SR_LOOKAHEAD_MAX_SAMPLES=2
export SR_LOOKAHEAD_CHUNK_SIZE=2

export TRAIN_DATASET=pminervini/HaluEval
export TRAIN_SUBSET=qa
export TRAIN_SPLIT=data
export EVAL_DATASET=drop

export MAX_TRAIN_SAMPLES=none
export MAX_EVAL_SAMPLES=none

export EVAL_EVERY_STEPS=50
export EVAL_START_STEP=100
export EVAL_LOSS=false

export TRAIN_EPOCHS=1
export TRAIN_MAX_STEPS=400
export TRAIN_EARLY_STOP_STEPS=151
export TRAIN_LR=5e-5
export TRAIN_WEIGHT_DECAY=0.0
export TRAIN_GRAD_ACCUM_STEPS=16

export ADD_HALLUCINATIONS=false

export OUTPUT_ROOT=outputs
export DEBUG=true
export TRAIN_TRACE_SAMPLES=3

export TRAIN_BATCH_SIZE=8
export EVAL_BATCH_SIZE=32

export CONTRASTIVE_BATCH_SIZE=8
export CONTRASTIVE_MAX_LENGTH=16384
export CONTRASTIVE_SHUFFLE=false
export CONTRASTIVE_DROP_LAST=false
export CONTRASTIVE_SEED=42
export CONTRASTIVE_PADDING=true
export CONTRASTIVE_TRUNCATION=true
export CONTRASTIVE_RETURN_TENSORS=none

# Visualization outputs for SR soft masking.
# soft_mask = metric^{-1}; smaller values indicate stronger suppression.
export SR_VISUALIZE_MASK=true
export SR_VISUALIZE_EVERY=50
export SR_VISUALIZE_MAX_ELEMENTS=4096
export SR_VISUALIZATION_DIR=outputs/visualization/soft_mask

# Optional debug output used by the ablation runner.
export SR_DEBUG=true
export SR_DEBUG_DIR=outputs/visualization/sr_element_metric

# Eval datasets for ablation runner. The runner overwrites EVAL_DATASET per run.
export ABLATION_EVAL_DATASETS="drop hotpotqa_fullwiki halueval_dialogue halueval_summarization"


# ---- Auto-added by ablation.sh ----
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export FROZEN_BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
export MODEL_TRUST_REMOTE_CODE=false
export REQUIRE_CHAT_TEMPLATE=false

# Keep training/hallucination judges fixed to Qwen; ablate only the eval judge.
export JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
export HALLUCINATION_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
export EVAL_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct

export EXPERIMENTS=contrastive_lora
export METHOD=contrastive_lora
export EVAL_DATASET=drop
export OUTPUT_ROOT=outputs/ablation_base_model/20260622_055843/model_meta_llama_Llama_3p1_8B_Instruct/judge_Qwen_Qwen2p5_7B_Instruct/seed_42/layer_single_27/maxstep_400/eval_drop/mask_backward/method_contrastive_lora
export DEBUG=true

export SR_DEBUG=true
export SR_DEBUG_DIR=outputs/ablation_base_model/20260622_055843/model_meta_llama_Llama_3p1_8B_Instruct/judge_Qwen_Qwen2p5_7B_Instruct/seed_42/layer_single_27/maxstep_400/eval_drop/mask_backward/method_contrastive_lora/debug/sr_element_metric
export SR_VISUALIZE_MASK=true
export SR_VISUALIZE_EVERY=50
export SR_VISUALIZE_MAX_ELEMENTS=4096
export SR_VISUALIZATION_DIR=outputs/visualization_ablation_base_model/20260622_055843/model_meta_llama_Llama_3p1_8B_Instruct/judge_Qwen_Qwen2p5_7B_Instruct/seed_42/layer_single_27/maxstep_400/eval_drop/mask_backward/method_contrastive_lora/soft_mask

export SR_MASK_PLACEMENT=backward
export SR_MASK_MODE=backward

export SR_RUN_NAME=ablation_base_model__model_meta_llama_Llama_3p1_8B_Instruct__judge_Qwen_Qwen2p5_7B_Instruct__seed_42__layer_single_27__eval_drop__mask_backward__method_contrastive_lora
export SR_RUN_GROUP=20260622_055843
export SR_ABLATION_EXPERIMENT=ablation_base_model
export SR_ABLATION_METHOD=contrastive_lora
export SR_ABLATION_EVAL_DATASET=drop
export SR_ABLATION_MASK_PLACEMENT=backward
export SR_ABLATION_LAYER_NAME=single_27
export SR_ABLATION_MODULE_NAME=mlp
export SR_ABLATION_MAX_STEP=400
export SR_ABLATION_OUTPUT_ROOT=outputs/ablation_base_model/20260622_055843/model_meta_llama_Llama_3p1_8B_Instruct/judge_Qwen_Qwen2p5_7B_Instruct/seed_42/layer_single_27/maxstep_400/eval_drop/mask_backward/method_contrastive_lora
export SR_ABLATION_VIS_DIR=outputs/visualization_ablation_base_model/20260622_055843/model_meta_llama_Llama_3p1_8B_Instruct/judge_Qwen_Qwen2p5_7B_Instruct/seed_42/layer_single_27/maxstep_400/eval_drop/mask_backward/method_contrastive_lora/soft_mask

export SR_TARGET_MODULES=gate_proj,up_proj,down_proj
export SR_TARGET_LAYERS=1
export SR_TARGET_LAYER_INDICES=27

export CONTRASTIVE_SEED=42
export TRAIN_SEED=42
export SEED=42
export PYTHONHASHSEED=42

export TRAIN_MAX_STEPS=400
export TRAIN_EARLY_STOP_STEPS=151
export EVAL_EVERY_STEPS=50
export EVAL_START_STEP=100
