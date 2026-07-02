set -e
set -x

# Run Anchored Learning after creating an SFT reference model.
# The student starts from BASE_MODEL. The teacher/reference points to p_sft.

DATA_JSON=${DATA_JSON:-/path/to/train.json}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
SFT_REFERENCE_MODEL=${SFT_REFERENCE_MODEL:-./output/anchored_learning_sft_reference}
OUTPUT_DIR=${OUTPUT_DIR:-./output/anchored_learning}

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 2"
OPTS+=" --learning_rate 1e-5"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 5"
OPTS+=" --save_path ${OUTPUT_DIR}"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --chunked_loss_size 2048"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path ${BASE_MODEL}"
OPTS+=" --teacher_name_or_path ${SFT_REFERENCE_MODEL}"
OPTS+=" --enable_thinking False"

# ============ DataArguments ============
OPTS+=" --train_dataset_path ${DATA_JSON}"
OPTS+=" --max_len 4096"
OPTS+=" --input_key instruction"
OPTS+=" --output_key output"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 8"

# ============ Anchored Learning ============
OPTS+=" --kd_algorithm anchored_learning"
OPTS+=" --kd_loss_fn anchored_kl"
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_temperature 1.0"
OPTS+=" --anchor_alpha 0.5"
OPTS+=" --anchor_interpolation logit"
OPTS+=" --teacher_tp_size 8"
OPTS+=" --teacher_mem_fraction_static 0.4"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb False"

python -m kdflow.cli.train_kd_off_policy $OPTS
