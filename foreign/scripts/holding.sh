#!/bin/bash

INTERVAL=1800 

is_gpu_idle() {
    local gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    for mem in $gpu_usage; do
        if [ "$mem" -ne 0 ]; then
            return 1
        fi
    done
    return 0
}

while true; do
    echo "Checking GPU status..."
    if is_gpu_idle; then
        echo "All GPUs are idle. Executing command: $CMD"
        CUDA_VISIBLE_DEVICES=7 python3 foreign/train_start.py \
            --num-gpus 1 --config-file configs/coco/fsod_r101_base_1shot_debug.yaml     \
            MODEL.WEIGHTS checkpoints/coco/defrcn_one/defrcn_det_r101_base/model_reset_remove.pth  \
            OUTPUT_DIR checkpoints/holding
    else
        echo "GPUs are busy. Retrying in $INTERVAL seconds..."
    fi
    sleep $INTERVAL
done
