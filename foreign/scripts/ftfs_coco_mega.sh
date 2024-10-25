#!/usr/bin/env bash
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO

EXPNAME=$2
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
#         --eval-only \
#         --num-gpus 7 --config-file configs/coco/coco_pretrain.yaml     \
#         MODEL.WEIGHTS checkpoints/coco/r101_base/model_final.pth \
#         OUTPUT_DIR ${SAVEDIR}/r101_base \

surgery(){
    python3 foreign/model_surgery.py --dataset coco --method remove                         \
        --src-path checkpoints/coco/base_r101/model_final.pth                        \
        --save-dir checkpoints/coco/base_r101
}


BASE_WEIGHT=checkpoints/coco/base_r101/model_reset_remove.pth


fs_base(){
# Test Base:
    for shot in 1 #1 3 5 10 30
    do
        for seed in 0 5 9
        do
            python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --suffix base
            CONFIG_PATH=configs/coco/fsod_r101_base_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/fsod_r101_base/fsrw-like/${shot}shot_seed${seed}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 1 --config-file ${CONFIG_PATH} \
                            MODEL.WEIGHTS ${BASE_WEIGHT} \
                            OUTPUT_DIR ${OUTPUT_DIR} \
                            MEGA.ENABLE True  \
                            MEGA.PHASE base_train
            rm $CONFIG_PATH
        done
    done
}

fs_novel(){
    for shot in 1 #1 3 5 10 30
    do
        for seed in 0 5 9
        do
            python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --suffix novel
            CONFIG_PATH=configs/coco/fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 --config-file ${CONFIG_PATH} \
                            MODEL.WEIGHTS ${BASE_WEIGHT} \
                            OUTPUT_DIR ${OUTPUT_DIR} \
                            SOLVER.CHECKPOINT_PERIOD 100
            rm $CONFIG_PATH
        done
    done
}

fs_base_eval(){
# Test Base:
    for shot in 1 #1 3 5 10 30
    do
        for seed in 0
        do
            python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --suffix base
            CONFIG_PATH=configs/coco/fsod_r101_base_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/fsod_r101_base/fsrw-like/${shot}shot_seed${seed}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 --config-file ${CONFIG_PATH} \
                            --eval-only \
                            MODEL.WEIGHTS ${SAVEDIR}/r101_base/model_final.pth \
                            OUTPUT_DIR ${OUTPUT_DIR}
            rm $CONFIG_PATH
        done
    done
}

fs_novel_eval(){
    for shot in 1 #1 3 5 10 30
    do
        for seed in 0
        do
            python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --suffix novel
            CONFIG_PATH=configs/coco/fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 --config-file ${CONFIG_PATH} \
                            --eval-only \
                            MODEL.WEIGHTS ${SAVEDIR}/r101_base/model_final.pth \
                            OUTPUT_DIR ${OUTPUT_DIR}\
                            SOLVER.CHECKPOINT_PERIOD 10000
            rm $CONFIG_PATH
        done
    done
}

case $1 in
    "fs_base")
        fs_base
        ;;
    "fs_novel")
        fs_novel
        ;;
    "fs_base_eval")
        fs_base_eval
        ;;
    "fs_novel_eval")
        fs_novel_eval
        ;;
    "sur")
        surgery
        ;;
    *)
        echo "Usage: $0 {fs_base|fs_novel}"
        exit 1
        ;;
esac