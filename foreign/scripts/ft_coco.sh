#!/usr/bin/env bash
export NCCL_DEBUG=WARN

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
#         --eval-only \
#         --num-gpus 7 --config-file configs/coco/coco_pretrain.yaml     \
#         MODEL.WEIGHTS checkpoints/coco/r101_base/model_final.pth \
#         OUTPUT_DIR ${SAVEDIR}/r101_base \


# python3 foreign/model_surgery.py --dataset coco --method remove                         \
#     --src-path ${SAVEDIR}/r101_base/model_final.pth                        \
#     --save-dir ${SAVEDIR}/r101_base

BASE_WEIGHT=checkpoints/coco/base_r101/model_reset_remove.pth
OUTPUT_DIR=${SAVEDIR}/ft_r101_novel

ft_novel(){
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 \
         --config-file configs/coco/coco_ft_novel.yaml \
            MODEL.WEIGHTS ${BASE_WEIGHT} \
            OUTPUT_DIR ${OUTPUT_DIR} \
            SOLVER.CHECKPOINT_PERIOD 500
}

ft_novel