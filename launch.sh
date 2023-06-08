export OMP_NUM_THREADS=12
BATCH_SIZE=80
torchrun --nproc_per_node=8 --master_port=8088 train.py \
    --trainer='zo' \
    --model_name_or_path openlm-research/open_llama_7b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --bf16_full_eval True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps=10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --dataloader_num_workers=10 \
