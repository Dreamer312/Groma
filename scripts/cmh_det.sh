#!/bin/bash

export NCCL_P2P_DISABLE=1
export PYTHONPATH=/file01/cmh/project/Groma
export CUDA_VISIBLE_DEVICES=0,1,2,4
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25002 \
     /file01/cmh/project/Groma/groma/train/train_det.py \
    --vis_encoder "/file01/cmh/project/Groma/pretrain_model/dinov2" \
    --dataset_config /file01/cmh/project/Groma/groma/data/configs/cmh_det.py \
    --bf16 True \
    --tf32 True \
    --num_classes 1 \
    --num_queries 300 \
    --two_stage True \
    --with_box_refine True \
    --ddetr_hidden_dim 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_feature_levels 1 \
    --freeze_vis_encoder True \
    --num_train_epochs 12 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 100 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --dataloader_num_workers 3 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --report_to none \
    --output_dir "/file01/cmh/project/Groma/det_debug_output" \
    | tee "/file01/cmh/project/Groma/det_debug_output"/train.log
