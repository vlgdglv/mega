#!/usr/bin/env bash
export NCCL_DEBUG=WARN

EXPNAME=$2
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=weights/ImageNetPretrained/MSRA/R-101.pkl                            # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

BASE_WEIGHT=checkpoints/coco/base_r101/model_reset_remove.pth

ft_base(){
   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 \
         --config-file configs/coco/coco_ft_base.yaml \
            MODEL.WEIGHTS ${BASE_WEIGHT} \
            OUTPUT_DIR ${SAVEDIR}/ft_r101_base  \
            SOLVER.CHECKPOINT_PERIOD 1000 
}

ft_novel(){
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 \
         --config-file configs/coco/coco_ft_novel.yaml \
            MODEL.WEIGHTS ${BASE_WEIGHT} \
            OUTPUT_DIR ${SAVEDIR}/ft_r101_novel \
            SOLVER.CHECKPOINT_PERIOD 1000
}

case $1 in
    "ft_base")
        ft_base
        ;;
    "ft_novel")
        ft_novel
        ;;
    *)
        echo "Usage: $0 {fs_base|fs_novel}"
        exit 1
        ;;
esac