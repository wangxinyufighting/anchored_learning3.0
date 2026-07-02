set -e
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Start ray before first running
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 8"
OPTS+=" --learning_rate 2e-6"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ./output/multi_teacher_off_policy_kd"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path Qwen3/Qwen3-4B"
OPTS+=" --enable_thinking False"

# ============ DataArguments ============
# The dataset must contain a `teacher_routing_key` field matching keys in teacher_config.json.
OPTS+=" --train_dataset_path /path/to/your/mixed/dataset"
OPTS+=" --max_len 4096"
OPTS+=" --prompt_max_len 2048"
OPTS+=" --generate_max_len 2048"
OPTS+=" --input_key conversations"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples True"
OPTS+=" --teacher_routing_key teacher_routing_key"

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm vanilla_kd"
OPTS+=" --teacher_dp_size 2"
OPTS+=" --teacher_tp_size 4"
OPTS+=" --teacher_mem_fraction_static 0.6"
OPTS+=" --teacher_forward_n_batches 10"
OPTS+=" --multi_teacher_config $SCRIPT_DIR/teacher_config.json"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 10"
OPTS+=" --use_wandb True"
OPTS+=" --wandb_project KDFlow"
OPTS+=" --wandb_group multi_teacher_off_policy_kd"
OPTS+=" --wandb_run_name qwen3_14b_to_4b_vanilla_kd"
OPTS+=" --wandb_mode offline"
OPTS+=" --wandb_dir ./output"

python -m kdflow.cli.train_kd_off_policy $OPTS
