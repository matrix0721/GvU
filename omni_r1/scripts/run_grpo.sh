#cd src/r1-v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"



torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    omni_r1/grpo.py \
    --deepspeed "omni_r1/configs/zero1.json" \
    --output_dir "outputs_dir" \
    --model_name_or_path "models/X-Omni-En" \
    --dataset_name "dataset_name" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_generations 2 \
    --beta 0.03 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 4 \
    --run_name gvu \
    --learning_rate 1e-6 \
    --save_steps 100 \
    --save_only_model true \
    --attn_implementation eager \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_namespan_exclude "['model.lm_embed_tokens', 'model.mm_embed_tokens' , 'embed_tokens','mm_head']"