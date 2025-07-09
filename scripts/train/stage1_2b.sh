#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-1}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export WANDB_PROJECT=videollama3_qwen2.5_2b
RUN_NAME=stage_1
DATA_DIR=DATASETS/STAGE1
OUTP_DIR=work_dirs

# python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.run --nnodes $WORLD_SIZE \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     --node_rank $RANK \
#     videollama3/train.py \
#     --deepspeed scripts/zero3.json \
#     --model_type videollama3_qwen2 \
#     --model_path Qwen/Qwen2.5-1.5B-Instruct \
#     --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
#     --mm_projector_type mlp2x_gelu \
#     --data_path ${DATA_DIR}/annotations.json \
#     --data_folder ${DATA_DIR} \
#     --image_merge_size 1 \
#     --video_merge_size 2 \
#     --fps 1 \
#     --max_frames 180 \
#     --model_max_length 6000 \
#     --mm_max_length 6000 \
#     --bf16 True \
#     --tf32 False \
#     --fp16 False \
#     --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $LOCAL_BATCH_SIZE \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 2 \
#     --mm_projector_lr 1e-3 \
#     --vision_encoder_lr 1e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 16 \
#     --report_to tensorboard \
#     --run_name $RUN_NAME

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero3.json \
    --model_type videollama3_qwen2 \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${DATA_DIR}/annotations.json \
    --data_folder ${DATA_DIR} \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 180 \
    --model_max_length 6000 \
    --mm_max_length 6000 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --mm_projector_lr 1e-3 \
    --vision_encoder_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name $RUN_NAME
