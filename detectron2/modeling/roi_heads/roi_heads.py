# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, cat, cross_entropy
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, ResNet
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNContrastOutputs
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head
from ..contrastives import (
    ContrastiveLoss,
    ContrastiveEncoder
)
from ..mega_model import ROIVAE

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
        only_sample_fg_proposals=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt
        self.only_sample_fg_proposals = only_sample_fg_proposals

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
            "only_sample_fg_proposals": cfg.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS,
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # only sample fg proposals to train recognition branch (ref to subsample_labels)
        if self.only_sample_fg_proposals:
            if has_gt:
                positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[0]
                num_pos = int(self.batch_size_per_image * self.positive_fraction)
                # protect against not enough positive examples
                num_pos = min(positive.numel(), num_pos)
                # randomly select positive and negative examples
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                sampled_idxs = positive[perm1]
            else:  # no gt, only keep 1 bg proposal to fill the slot
                sampled_idxs = torch.zeros_like(matched_idxs[0:1])
            return sampled_idxs, gt_classes[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt and self.training: # DEBUG
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            ) # (N, M)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # assign the iou
            iou, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = iou[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        try:
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        except:
            pass
        #print("num_fg: {}; num_bg: {}".format(num_fg_samples, num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        learner: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head
        self.learner = learner

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        if cfg.MEGA.ROIHEADS_ENABLE:
            learner = ROIVAE(input_channels=input_shape[in_features[0]].channels,
                             output_channels=out_channels,
                             latent_dim=cfg.MEGA.ROIHEADS_LEARNER_DIM, 
                             phase=cfg.MEGA.PHASE,
                             recon_weight=cfg.MEGA.ROIHEADS_RECON_WEIGHT, 
                             reg_weight=cfg.MEGA.ROIHEADS_REG_WEIGHT)
        else:
            learner = None
        ret["learner"] = learner
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        """
        boxes: List[BoxList], torch.Size([NUM_BOX, 4]) * BS
        features[0] torch.Size([BS, 1024, H, W])
        pooled_x: torch.Size([BS * NUM_BOX, 1024, 7, 7])
        resed_x: torch.Size([BS * NUM_BOX, 2048, 4, 4])
        """
        h = self.pooler(features, boxes)
        x = self.res5(h)
        if self.learner is not None:
            return x, h
        else:
            return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        """
        features['res4']: torch.Size([BS, 1024, 50, 75])
        box_features: torch.Size([BS * BOX_NUM, 2048, 4, 4])
        """
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        if self.learner is not None:
            box_features, before_features = box_features
            NUM_BOX = proposal_boxes[0].tensor.shape[0]
            BSBBOX, C, H, W = before_features.shape
            BS = BSBBOX // NUM_BOX
            before_features = before_features.view(BS, NUM_BOX, C, H, W)
            before_features = before_features.mean(dim=1)
            BSBBOX, C, H, W = box_features.shape
            after_features = box_features.view(BS, NUM_BOX, C, H, W)
            after_features = after_features.mean(dim=1)
            learner_loss = self.learner.forward_and_loss(before_features, after_features) 
        predictions = self.box_predictor(box_features.mean(dim=[2, 3])) # default OPTION

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            if self.learner is not None:
                losses.update({"learner_loss": learner_loss})
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        learner: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        self.learner = learner

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        # if cfg.MEGA.ROIHEADS_ENABLE:
        #     learner = None
        # else:
        learner = None
        ret["learner"] = learner
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        
        predictions = self.box_predictor(box_features)
        box_features_flat = box_features.view(box_features.size(0), -1)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)

            if self.learner is not None:
                learner_loss = self.learner.update(self.box_head)
                losses["roi_learner_loss"] = learner_loss

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)

class Localizer(nn.Module):
    def __init__(self, in_channels: int, pooler_resolution: int, fc_dim: int, num_classes: int, box_dim: int =4):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.localizer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pooler_resolution * pooler_resolution, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, box_dim * num_classes),
        )

    def forward(self, features):
        return self.localizer(features)


@ROI_HEADS_REGISTRY.register()
class VAEROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        localizer: nn.Module,
        vae_cls: nn.Module,
        input_shape: ShapeSpec,
        box_predictor: Optional[nn.Module] = None,
        mask_head: Optional[nn.Module] = None,
        contrastive_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            featurer (nn.Sequential): some model
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        # if isinstance(res5, (list, tuple)):
        #     res5 = nn.Sequential(*res5)
        self.vae_cls = vae_cls
        self.localizer = localizer

        self.box_predictor = box_predictor
        for param in self.box_predictor.parameters():
            param.requires_grad = False
        self.mask_on = mask_head is not None
        self.contrastive_head = contrastive_head
         

        if self.mask_on:
            self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON

        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        roi_out_dim=cfg.MEGA.ROIHEADS_OUT_DIM
        ret["vae_cls"] = VAEBlock(
            in_channels=input_shape[in_features[0]].channels, 
            out_dim=roi_out_dim,
            latent_dim=cfg.MEGA.VAE_LATENT_DIM, 
            pooler_resolution=pooler_resolution,
            num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )

        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=roi_out_dim, height=1, width=1)
        )

        # ret["contrastive_head"] = ContrastiveLoss(
        #     cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE,
        #     cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD,
        #     cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC
        # )
        ret["localizer"] = Localizer(in_channels=input_shape[in_features[0]].channels, 
                                     pooler_resolution=pooler_resolution, 
                                     fc_dim=cfg.MODEL.ROI_BOX_HEAD.FC_DIM, 
                                     num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,)
        ret["input_shape"] = input_shape

        return ret

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        """
        features['res4']: torch.Size([BS, 1024, 50, 75])
        box_features: torch.Size([BS * BOX_NUM, 2048, 4, 4])
        """
        proposal_boxes = [x.proposal_boxes for x in proposals]
        # box_features = self._shared_roi_transform(
        #     [features[f] for f in self.in_features], proposal_boxes
        # )

        # features_in: [BS, 1024, H, W] -> pooled_features: [BS * BOX_NUM, 1024, 7, 7]
        pooled_features = self.pooler([features[f] for f in self.in_features], proposal_boxes) 
        
        returned_things = self.vae_cls(pooled_features)
        proposal_deltas = self.localizer(pooled_features)
        
        # pooled_features: [BS * BOX_NUM, 1024, 7, 7] -> box_features: [BS * BOX_NUM, ?]
        # returned_things = self.featurer(pooled_features, self.training)
        if self.training:
            scores, decoded, mu, logvar, z = returned_things
        else:
            scores = returned_things

        # box_features: [BS * BOX_NUM, ?]
        # predictions = self.box_predictor(box_features) # default OPTION

        if self.training:
            del features
            losses = self.box_predictor.losses((scores, proposal_deltas), proposals)
            kl_loss = self.vae_cls.kl_loss(mu, logvar)
            losses['roi_kl_loss'] = kl_loss
            recon_loss = self.vae_cls.recon_loss(decoded, pooled_features)
            losses['roi_recon_loss'] = recon_loss

            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )
            ious = (
                cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0.0)
            )
            # if self.contrastive_head is not None:
            #     con_loss = self.contrastive_head(z, gt_classes, ious)
            #     losses['roi_contrastive_loss'] = con_loss

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            # kl_loss =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # losses['roi_kl_loss'] = kl_loss
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference((scores, proposal_deltas), proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
        
        # loss weights
        if self.cls_loss_weight is not None and self.cls_loss_weight.device != scores.device:
            self.cls_loss_weight = self.cls_loss_weight.to(scores.device)
        if self.focal_scaled_loss is not None:
            loss_cls = self.focal_loss(scores, gt_classes, gamma=self.focal_scaled_loss)
        else:    
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean") if self.cls_loss_weight is None else \
                       cross_entropy(scores, gt_classes, reduction="mean", weight=self.cls_loss_weight)
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


def _log_classification_stats(pred_logits, gt_classes, prefix=""):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    try:
        storage = get_event_storage()
        storage.put_scalar(f"cls_acc", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar(f"fg_cls_acc", fg_num_accurate / num_fg)
            storage.put_scalar(f"false_neg_ratio", num_false_negative / num_fg)
            #print("cls_accuracy {:.2f}; fg_cls_accuracy {:.2f}; false_negative {:.2f}".format(num_accurate / num_instances, fg_num_accurate / num_fg, num_false_negative / num_fg))
    except:
        pass

class VAEBlock(nn.Module):
    def __init__(self, in_channels, out_dim,  latent_dim, pooler_resolution, num_classes, conv_dim=256, hidden_dim=512, prior_type='normal'):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.pooler_resolution = pooler_resolution
        
        org_dim = in_channels * pooler_resolution * pooler_resolution
        # Encoder: [BS * NUM, C_in, 7, 7]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True), # [BS * NUM, C_in, 4, 4], floor((7 + 2*1 - 3)/2 + 1) = 4,
            nn.BatchNorm2d(in_channels),
            # nn.Flatten(start_dim=1)
            nn.Conv2d(in_channels, conv_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True), # floor((4 + 2*1 - 3)/1 + 1) = 4
            nn.BatchNorm2d(conv_dim),
        ) 
        # encoder_out_dim = in_channels * conv_dim * 4 * 4
        self.fc_mu = nn.Linear(conv_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_dim, latent_dim)
        
        # self.decoder = nn.Linear(latent_dim, out_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, org_dim),
            nn.ReLU(),
        )
        # [BS * NUM, C_in * 2, 4, 4]
        # self.encoder = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(in_channels * self.pooler_resolution * self.pooler_resolution, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )

        # self.channelwise_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True), # [BS * NUM, C_in, 7, 7], floor((7 + 2*0 - 1)/1 + 1) = 7
        #     nn.BatchNorm2d(in_channels),
        #     nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(inplace=True), # floor((7 + 2*1 - 3)/2 + 1) = 4
        #     nn.BatchNorm2d(in_channels * 2),
        # ) # [BS * NUM, C_in * 2, 4, 4]

        # reduced_res = self.pooler_resolution // 4

        # Linear decoder

        self.cls_score = nn.Linear(org_dim, num_classes + 1) # one background class (hence + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x, is_train=True):
        # Encode
        encoded = self.encoder(x)
        # channel_encoded = self.channelwise_encoder(x)
        # encoded = conv_encoded + channel_encoded

        # encoded = torch.flatten(encoded, start_dim=1)
        encoded = encoded.mean([2, 3])
        # Compute mu and logvar
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        decoded = self.decoder(z)

        scores = self.cls_score(decoded)
        if is_train:
            return scores, decoded, mu, logvar, z
        else:
            return scores

    def kl_loss(self, mu, logvar):
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return  kl_loss
    
    def recon_loss(self, decoded, x):
        # Reconstruction loss
        BS, C, H, W = x.shape
        recon_loss = F.mse_loss(decoded, x.view(BS, -1))
        return recon_loss

@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, 
                 *,
                 weight_decay,
                 decay_steps,
                 decay_rate, 
                 encoder,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.encoder = encoder


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        encoder = ContrastiveEncoder(
            cfg.MODEL.ROI_BOX_HEAD.FC_DIM, 
            cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        )
        ret["weight_decay"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        ret["decay_steps"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        ret["decay_rate"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE
        ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        ret["encoder"] = encoder
        box_predictor = FastRCNNContrastOutputs(cfg, ret["box_head"].output_shape)
        ret["box_predictor"] = box_predictor
        return ret


    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        predictions = self.box_predictor(box_features)
        box_features_contrast = self.encoder(box_features)
        del box_features

        if self.weight_decay:
            storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.box_predictor.loss_weight["loss_contrast"] *= self.decay_rate

        # outputs = FastRCNNContrastOutputs(
        #     self.box2box_transform,
        #     pred_class_logits,
        #     pred_proposal_deltas,
        #     proposals,
        #     self.smooth_l1_beta,
        #     box_features_contrast,
        #     self.criterion,
        #     self.contrast_loss_weight,
        #     self.box_reg_weight,
        #     self.cl_head_only,
        # )
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, box_features_contrast)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
        