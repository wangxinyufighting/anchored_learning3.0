set -e
set -x

# First train the fixed SFT reference model p_sft used by Anchored Learning.
# Replace DATA_JSON and model paths with your local values.

DATA_JSON=${DATA_JSON:-/path/to/train.json}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
SFT_REFERENCE_OUT=${SFT_REFERENCE_OUT:-./output/anchored_learning_sft_reference}

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 2"
OPTS+=" --learning_rate 1e-5"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ${SFT_REFERENCE_OUT}"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path ${BASE_MODEL}"
OPTS+=" --enable_thinking False"

# ============ DataArguments ============
OPTS+=" --train_dataset_path ${DATA_JSON}"
OPTS+=" --max_len 4096"
OPTS+=" --input_key instruction"
OPTS+=" --output_key output"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 8"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb False"

torchrun --nproc_per_node=8 -m kdflow.cli.train_sft $OPTS
