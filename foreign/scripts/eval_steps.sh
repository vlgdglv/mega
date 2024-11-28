#!/bin/bash

SEARCH_PATH="checkpoints/coco/exp1128/steps"

python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
            --shot 1 --seed 0 --suffix novel
CONFIG_PATH=configs/coco/fsod_r101_novel_${shot}shot_seed${seed}.yaml
OUTPUT_DIR=${SEARCH_PATH}/step_result
mkdir -p $OUTPUT_DIR

for file in $(find "$SEARCH_PATH" -type f -name "*.pth"); do
    echo "Found .pth file: $file"
    BASE_NAME=$(basename "$file" .pth)
    NAME=$(echo "$BASE_NAME" | cut -d'.' -f2-)  

    RESULT_FILE="result_${NAME}.txt"
    echo "Result file: $RESULT_FILE"

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
        --eval-only \
        --num-gpus 7 --config-file $CONFIG_PATH   \
        MODEL.WEIGHTS $file \
        OUTPUT_DIR checkpoints/coco/exp1126/r101_base/eval | tee $OUTPUT_DIR/$RESULT_FILE
done


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
#         --eval-only \
#         --num-gpus 7 --config-file configs/coco/coco_pretrain.yaml     \
#         MODEL.WEIGHTS checkpoints/coco/exp1126/r101_base/model_0089999.pth \
#         OUTPUT_DIR checkpoints/coco/exp1126/r101_base/eval 