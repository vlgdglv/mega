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
        CUDA_VISIBLE_DEVICES=0 python3 foreign/train_start.py \
            --num-gpus 1 --config-file configs/coco/fsod_r101_base_1shot_debug.yaml     \
            MODEL.WEIGHTS weights/ImageNetPretrained/MSRA/R-101.pkl  \
            OUTPUT_DIR checkpoints/holding \
            MEGA.ENABLE_GRADIENT_SCALE True \
            MEGA.RPN_GRADIENT_SCALE 0.0 \
            MEGA.ROIHEADS_GRADIENT_SCALE 0.1 \
            SOLVER.MAX_ITER 500000

    else
        echo "GPUs are busy. Retrying in $INTERVAL seconds..."
    fi
    sleep $INTERVAL
done
