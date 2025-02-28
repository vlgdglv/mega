#!/usr/bin/env bash
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL 

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
        --src-path ${SAVEDIR}/ft_r101_base/model_final.pth                        \
        --save-dir ${SAVEDIR}/ft_r101_base
}


BASE_WEIGHT=checkpoints/coco/base_r101/model_reset_remove.pth
ORI_WEIGHT=checkpoints/coco/base_r101/model_final.pth

ft_base(){
# Test Base:
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
        --num-gpus 7 --config-file configs/coco/contrastive/coco_ft_base.yaml \
            MODEL.WEIGHTS ${ORI_WEIGHT} \
            OUTPUT_DIR ${SAVEDIR}/ft_r101_base \
            MEGA.ENABLE False  \
            MODEL.ROI_HEADS.NAME ContrastiveROIHeads \
            MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY False \
            MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.5 \
            
    python3 foreign/model_surgery.py --dataset coco --method remove \
        --src-path ${SAVEDIR}/ft_r101_base/model_final.pth \
        --save-dir ${SAVEDIR}/ft_r101_base
}

train_learner(){
# Test Base:
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py \
        --num-gpus 7 --config-file configs/coco/coco_learner.yaml \
            MODEL.WEIGHTS ${ORI_WEIGHT} \
            OUTPUT_DIR ${SAVEDIR}/ft_r101_base \
            MEGA.ENABLE True  \
            MEGA.PHASE base_train
    python3 foreign/model_surgery.py --dataset coco --method remove                         \
        --src-path ${SAVEDIR}/ft_r101_base/model_final.pth                        \
        --save-dir ${SAVEDIR}/ft_r101_base
}

fs_novel(){
    for shot in 1 #1 3 5 10 30
    do
        for seed in 0 # 5 9
        do
            python3 foreign/create_config.py --dataset coco14 --config_root configs/coco \
                    --shot ${shot} --seed ${seed} --suffix novel
            CONFIG_PATH=configs/coco/fsod_r101_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/fsod_r101_novel/fsrw-like/${shot}shot_seed${seed}
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 foreign/train_start.py --num-gpus 7 --config-file ${CONFIG_PATH} \
                            MODEL.WEIGHTS checkpoints/coco/${EXPNAME}/ft_r101_base/model_reset_remove.pth \
                            OUTPUT_DIR ${OUTPUT_DIR} \
                            MEGA.ENABLE False \
                            MEGA.PHASE novel_train \
                            MEGA.ROIHEADS_GUIDE_WEIGHT 10.0 \
                            MEGA.RPN_GUIDE_WEIGHT 100.0 \
                            MEGA.ROIHEADS_LEARNER_DIM 512 \
                            # MODEL.ROI_HEADS.NAME ContrastiveROIHeads \
                            # MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY False \
                            # MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.5 \
                            # MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.0 
            rm $CONFIG_PATH
        done
    done
}

case $1 in
    "ft_base")
        ft_base
        ;;
    "fs_novel")
        fs_novel
        ;;
    "train_learner")
        train_learner
        ;;
    "sur")
        surgery
        ;;
    *)
        echo "Usage: $0 {ft_base|fs_novel}"
        exit 1
        ;;
esac