#!/usr/bin/env bash
export NCCL_DEBUG=WARN

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 python3 foreign/train_start.py \
        --num-gpus 8 --config-file configs/coco/coco_pretrain.yaml     \
        MODEL.WEIGHTS $IMAGENET_PRETRAIN \
        OUTPUT_DIR ${SAVEDIR}/r101_base \


python3 foreign/model_surgery.py --dataset coco --method remove                         \
    --src-path ${SAVEDIR}/r101_base/model_final.pth                        \
    --save-dir ${SAVEDIR}/r101_base

BASE_WEIGHT=${SAVEDIR}/r101_base/model_reset_remove.pth


# for shot in 30 #1 3 5 10 30
# do
#         for seed in 0
#         do
#                 python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
#                         --shot ${shot} --seed ${seed} --setting 'fsod'
#                 CONFIG_PATH=configs/coco/fsod_r101_novel_${shot}shot_seed${seed}.yaml
#                 OUTPUT_DIR=${SAVEDIR}/fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}
#                 python3 foreign/train_start.py --num-gpus 7 --config-file ${CONFIG_PATH} \
#                                 MODEL.WEIGHTS ${BASE_WEIGHT} \
#                                 OUTPUT_DIR ${OUTPUT_DIR}
#         done
# done
