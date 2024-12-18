export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
project_dir=$(grep "PROJECT_DIR" ../config.py | cut -d "'" -f 2)

BASE_MODEL="../llm/Mistral-7B-Instruct-v0.3"
TRAIN_DATA_PATH="../training_data/msmarco_gpt-4o-2024-08-06_bm25-top100-slidingwindow_9passes.jsonl"  # Train Dataset --> Hugging Face dataset or Local dataset
OUTPUT_DIR="../trained_models/RankMistral100"  # Directory to save the trained model
TOKENIZERS_PARALLELISM=True accelerate launch --config_file "train_configs/accel_config_deepspeed.yaml" --main_process_port 41021 --num_processes 4 "run_train.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --train_dataset_path "${TRAIN_DATA_PATH}" \
    --noisy_embedding_alpha 5 \
    --seed 42 \
    --max_passage_len 100 \
    --prompt_mode rank_GPT \
    --weighted_loss True \
    --variable_passages True \
    --do_train \
    --bf16 \
    --num_train_epochs 4 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --remove_unused_columns False \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --logging_steps 3 \
    --save_total_limit 10 \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir "${OUTPUT_DIR}" \
    --report_to none \

