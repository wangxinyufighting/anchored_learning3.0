set -e
set -x

# End-to-end Anchored Learning training script.
# Data format:
# [
#   {"instruction": "...", "input": "...", "output": "..."},
#   {"instruction": "...", "input": "...", "output": "..."}
# ]
#
# Usage:
#   DATA_JSON=/path/to/train.json \
#   BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
#   bash examples/anchored_learning/train_anchored_learning.sh
#
# To reuse an existing SFT reference:
#   RUN_REFERENCE_SFT=false \
#   SFT_REFERENCE_MODEL=/path/to/sft_reference \
#   DATA_JSON=/path/to/train.json \
#   BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
#   bash examples/anchored_learning/train_anchored_learning.sh

if [ -z "${DATA_JSON:-}" ]; then
  echo "DATA_JSON must point to your instruction/input/output JSON file."
  exit 1
fi

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output/anchored_learning}
SFT_REFERENCE_MODEL=${SFT_REFERENCE_MODEL:-${OUTPUT_ROOT}/sft_reference}
ANCHOR_OUTPUT_DIR=${ANCHOR_OUTPUT_DIR:-${OUTPUT_ROOT}/anchored_model}
RUN_REFERENCE_SFT=${RUN_REFERENCE_SFT:-true}
KDFLOW_RAY_ADDRESS=${KDFLOW_RAY_ADDRESS:-local}
export KDFLOW_RAY_ADDRESS

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-2}
MAX_LEN=${MAX_LEN:-4096}
PREPROCESS_NUM_WORKERS=${PREPROCESS_NUM_WORKERS:-8}
CHUNKED_LOSS_SIZE=${CHUNKED_LOSS_SIZE:-2048}

SFT_LR=${SFT_LR:-1e-5}
SFT_EPOCHS=${SFT_EPOCHS:-1}
ANCHOR_LR=${ANCHOR_LR:-1e-5}
ANCHOR_EPOCHS=${ANCHOR_EPOCHS:-5}
ANCHOR_ALPHA=${ANCHOR_ALPHA:-0.5}
ANCHOR_INTERPOLATION=${ANCHOR_INTERPOLATION:-logit}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-${NUM_GPUS_PER_NODE}}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.4}

COMMON_DATA_OPTS=""
COMMON_DATA_OPTS+=" --train_dataset_path ${DATA_JSON}"
COMMON_DATA_OPTS+=" --max_len ${MAX_LEN}"
COMMON_DATA_OPTS+=" --input_key instruction"
COMMON_DATA_OPTS+=" --output_key output"
COMMON_DATA_OPTS+=" --apply_chat_template True"
COMMON_DATA_OPTS+=" --preprocess_num_workers ${PREPROCESS_NUM_WORKERS}"

COMMON_TRAIN_OPTS=""
COMMON_TRAIN_OPTS+=" --num_nodes ${NUM_NODES}"
COMMON_TRAIN_OPTS+=" --num_gpus_per_node ${NUM_GPUS_PER_NODE}"
COMMON_TRAIN_OPTS+=" --backend fsdp2"
COMMON_TRAIN_OPTS+=" --train_batch_size ${TRAIN_BATCH_SIZE}"
COMMON_TRAIN_OPTS+=" --micro_train_batch_size ${MICRO_TRAIN_BATCH_SIZE}"
COMMON_TRAIN_OPTS+=" --lr_warmup_ratio 0.05"
COMMON_TRAIN_OPTS+=" --bf16 True"
COMMON_TRAIN_OPTS+=" --gradient_checkpointing True"

COMMON_LOG_OPTS=""
COMMON_LOG_OPTS+=" --logging_steps 10"
COMMON_LOG_OPTS+=" --use_wandb False"

if [ "${RUN_REFERENCE_SFT}" = "true" ]; then
  SFT_OPTS="${COMMON_TRAIN_OPTS}"
  SFT_OPTS+=" --learning_rate ${SFT_LR}"
  SFT_OPTS+=" --num_epochs ${SFT_EPOCHS}"
  SFT_OPTS+=" --save_path ${SFT_REFERENCE_MODEL}"
  SFT_OPTS+=" --student_name_or_path ${BASE_MODEL}"
  SFT_OPTS+=" --enable_thinking False"
  SFT_OPTS+="${COMMON_DATA_OPTS}"
  SFT_OPTS+="${COMMON_LOG_OPTS}"

  torchrun --nproc_per_node=${NUM_GPUS_PER_NODE} -m kdflow.cli.train_sft ${SFT_OPTS}
fi

ANCHOR_OPTS="${COMMON_TRAIN_OPTS}"
ANCHOR_OPTS+=" --learning_rate ${ANCHOR_LR}"
ANCHOR_OPTS+=" --num_epochs ${ANCHOR_EPOCHS}"
ANCHOR_OPTS+=" --save_path ${ANCHOR_OUTPUT_DIR}"
ANCHOR_OPTS+=" --chunked_loss_size ${CHUNKED_LOSS_SIZE}"
ANCHOR_OPTS+=" --student_name_or_path ${BASE_MODEL}"
ANCHOR_OPTS+=" --teacher_name_or_path ${SFT_REFERENCE_MODEL}"
ANCHOR_OPTS+=" --enable_thinking False"
ANCHOR_OPTS+="${COMMON_DATA_OPTS}"
ANCHOR_OPTS+=" --kd_algorithm anchored_learning"
ANCHOR_OPTS+=" --kd_loss_fn anchored_kl"
ANCHOR_OPTS+=" --kd_ratio 1.0"
ANCHOR_OPTS+=" --kd_temperature 1.0"
ANCHOR_OPTS+=" --anchor_alpha ${ANCHOR_ALPHA}"
ANCHOR_OPTS+=" --anchor_interpolation ${ANCHOR_INTERPOLATION}"
ANCHOR_OPTS+=" --teacher_tp_size ${TEACHER_TP_SIZE}"
ANCHOR_OPTS+=" --teacher_mem_fraction_static ${TEACHER_MEM_FRACTION_STATIC}"
ANCHOR_OPTS+="${COMMON_LOG_OPTS}"

python -m kdflow.cli.train_kd_off_policy ${ANCHOR_OPTS}
