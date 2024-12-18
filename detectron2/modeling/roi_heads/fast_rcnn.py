# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.layers.soft_nms import batched_soft_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from ..contrastives import (
    ContrastiveLoss,
    ContrastiveEncoder
)

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        soft_nms_enabled (bool): Indicate to use soft non-maximum suppression.
        soft_nms_method: (str): One of ['gaussian', 'linear', 'hard']
        soft_nms_sigma: (float): Sigma for gaussian soft nms. Value in (0, inf)
        soft_nms_prune: (float): Threshold for pruning during soft nms. Value in [0, 1]
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, 
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune, topk_per_image, s_bf_per_img, vis
        )
        for scores_per_image, boxes_per_image, image_shape, s_bf_per_img in zip(scores, boxes, image_shapes, scores_bf_multiply)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


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


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    soft_nms_enabled: bool,
    soft_nms_method: str,
    soft_nms_sigma: float,
    soft_nms_prune: float,
    topk_per_image: int,
    scores_bf_multiply: List[torch.Tensor],
    vis=False,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        scores_bf_multiply = scores_bf_multiply[valid_mask]

    scores = scores[:, :-1]
    scores_bf_multiply = scores_bf_multiply[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    scores_bf_multiply = scores_bf_multiply[filter_mask]

    # 2. Apply NMS for each class independently.
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
        )
        scores[keep] = soft_nms_scores   
        # scores_bf_multiply? (TBD)
        scores_bf_multiply = scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    scores_bf_multiply = scores_bf_multiply[keep]
    # here, create instances
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes) # [100x4]
    result.scores = scores # [100]
    if vis: # visualization: convert to the original scores before multiplying RPN scores
        result.scores = scores_bf_multiply         
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def softmax_cross_entropy_loss(self):
        """
        Deprecated
        """
        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Deprecated
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        soft_nms_enabled=False,
        soft_nms_method="gaussian",
        soft_nms_sigma=0.5,
        soft_nms_prune=0.001,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        clip_cls_emb: tuple = (False, None),
        no_box_delta: bool = False,
        bg_cls_loss_weight: None,
        multiply_rpn_score: tuple = (False, False),
        openset_test: None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_prune = soft_nms_prune
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        # RegionCLIP
        self.num_classes = num_classes
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.input_size = input_size
        self.use_clip_cls_emb = clip_cls_emb[0]
        if self.use_clip_cls_emb: # use CLIP text embeddings as classifier's weights
            input_size = clip_cls_emb[3] if clip_cls_emb[2] in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] else input_size
            text_emb_require_grad = False
            self.use_bias = False
            self.temperature = openset_test[2] # 0.01 is default for CLIP

            # class embedding
            self.cls_score = nn.Linear(input_size, num_classes, bias=self.use_bias)  
            with torch.no_grad():
                if clip_cls_emb[1] is not None: # it could be None during region feature extraction
                    pre_computed_w = torch.load(clip_cls_emb[1])  # ! weight loading [num_classes, 1024] for RN50
                    self.cls_score.weight.copy_(pre_computed_w)
                self.cls_score.weight.requires_grad = text_emb_require_grad # freeze embeddings
                if self.use_bias:
                    nn.init.constant_(self.cls_score.bias, 0)
            
            # background embedding
            self.cls_bg_score = nn.Linear(input_size, 1, bias=self.use_bias)  
            with torch.no_grad():
                nn.init.constant_(self.cls_bg_score.weight, 0)  # zero embeddings
                self.cls_bg_score.weight.requires_grad = text_emb_require_grad
                if self.use_bias:
                    nn.init.constant_(self.cls_bg_score.bias, 0)

            # class embedding during test 
            self.test_cls_score = None
            if openset_test[1] is not None:  # openset test enabled
                pre_computed_w = torch.load(openset_test[1])  # ! weight loading [#openset_test_num_cls, 1024] for RN50
                self.openset_test_num_cls = pre_computed_w.size(0)
                self.test_cls_score = nn.Linear(input_size, self.openset_test_num_cls, bias=self.use_bias)  
                self.test_cls_score.weight.requires_grad = False # freeze embeddings
                with torch.no_grad():
                    self.test_cls_score.weight.copy_(pre_computed_w)
                    if self.use_bias:
                        nn.init.constant_(self.test_cls_score.bias, 0)    
        else: # regular classification layer  
            self.cls_score = nn.Linear(input_size, num_classes + 1) # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
 
        # box regression layer
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim) #! only thing that gets finetuned!
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        # training options
        self.cls_loss_weight = None
        if bg_cls_loss_weight is not None:  # loss weigh for bg class
            self.cls_loss_weight = torch.ones(num_classes + 1)
            self.cls_loss_weight[-1] = bg_cls_loss_weight
        self.focal_scaled_loss = openset_test[3]  # focal scaling
        # inference options
        self.no_box_delta = no_box_delta  # box delta after regression
        self.multiply_rpn_score = multiply_rpn_score[0]
        self.vis = multiply_rpn_score[1] # if enabled, visualize scores before multiplying RPN scores
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        # if cfg.MODEL.CLIP.CROP_REGION_TYPE == "RPN":
        #     assert cfg.MODEL.CLIP.NO_BOX_DELTA is False
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "soft_nms_enabled"      : cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            "soft_nms_method"       : cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD,
            "soft_nms_sigma"        : cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA,
            "soft_nms_prune"        : cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # RegionCLIP
            "clip_cls_emb"          : (cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER, cfg.MODEL.CLIP.TEXT_EMB_PATH, cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.CLIP.TEXT_EMB_DIM),
            "no_box_delta"          : cfg.MODEL.CLIP.NO_BOX_DELTA or cfg.MODEL.CLIP.CROP_REGION_TYPE == 'GT',
            "bg_cls_loss_weight"    : cfg.MODEL.CLIP.BG_CLS_LOSS_WEIGHT,
            "multiply_rpn_score"    : (cfg.MODEL.CLIP.MULTIPLY_RPN_SCORE, cfg.MODEL.CLIP.VIS),
            "openset_test"          : (cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES, cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH, \
                                       cfg.MODEL.CLIP.CLSS_TEMP, cfg.MODEL.CLIP.FOCAL_SCALED_LOSS)
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        # use clip text embeddings as classifier's weights
        if self.use_clip_cls_emb: 
            normalized_x = F.normalize(x, p=2.0, dim=1)
             # open-set inference enabled
            if not self.training and self.test_cls_score is not None: 
                cls_scores = normalized_x @ F.normalize(self.test_cls_score.weight, p=2.0, dim=1).t()
                if self.use_bias:
                    cls_scores += self.test_cls_score.bias
            # training or closed-set model inference
            else: 
                cls_scores = normalized_x @ F.normalize(self.cls_score.weight, p=2.0, dim=1).t()
                if self.use_bias:
                    cls_scores += self.cls_score.bias
            
            # background class (zero embeddings)
            bg_score = self.cls_bg_score(normalized_x)
            if self.use_bias:
                bg_score += self.cls_bg_score.bias

            scores = torch.cat((cls_scores, bg_score), dim=1) # 1000x66
            scores = scores / self.temperature
        # regular classifier
        else:  
            scores = self.cls_score(x)
        
        # box regression
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

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

    def focal_loss(self, inputs, targets, gamma=0.5, reduction="mean"):
        """Inspired by RetinaNet implementation"""
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient
        
        # focal scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        loss = ce_loss * ((1 - p_t) ** gamma)

        # bg loss weight
        if self.cls_loss_weight is not None:
            loss_weight = torch.ones(loss.size(0)).to(p.device)
            loss_weight[targets == self.num_classes] = self.cls_loss_weight[-1].item()
            loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        # optional: multiply class scores with RPN scores 
        scores_bf_multiply = scores  # as a backup for visualization purpose
        if self.multiply_rpn_score and not self.training:
            rpn_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * rpn_s[:, None]) ** 0.5 for s, rpn_s in zip(scores, rpn_scores)]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.soft_nms_enabled,
            self.soft_nms_method,
            self.soft_nms_sigma,
            self.soft_nms_prune,
            self.test_topk_per_image,
            scores_bf_multiply = scores_bf_multiply,
            vis = True if self.vis else False,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        # don't apply box delta, such as GT boxes
        if self.no_box_delta:
            predict_boxes = proposal_boxes
        # apply box delta
        else:
            predict_boxes = self.box2box_transform.apply_deltas(
                proposal_deltas,
                proposal_boxes,
            )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


class FastRCNNContrastOutputs(FastRCNNOutputLayers):
    """
    Add a multi-task contrastive loss branch for FastRCNNOutputs
    """
    @configurable
    def __init__(
        self,
        *,
        cl_head_only,
        constrastive_criterion,
        use_cossim,
        cossim_scale,
        cls_use_bias,
        **kwargs,
    ):
        """
        Args:
            box_cls_feat_con (Tensor): the projected features
                to calculate supervised contrastive loss upon
            criterion (ContrastiveLoss <- nn.Module): ContrastiveLoss is implemented in fsdet/modeling/contrastive_loss.py
        """
        super().__init__(**kwargs)

        self.constrastive_criterion = constrastive_criterion
        self.cl_head_only = cl_head_only
        self.use_cossim = use_cossim
        self.cossim_scale = cossim_scale

        if not cls_use_bias:
            self.cls_score = nn.Linear(self.input_size, self.num_classes + 1, bias=False) # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)

        if self.cl_head_only:
            for param in self.cls_score.parameters():
                param.requires_grad = False
            for param in self.bbox_pred.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        constrastive_criterion = ContrastiveLoss(
            cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE,
            cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD,
            cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC
        )
        ret["cl_head_only"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        ret["constrastive_criterion"] = constrastive_criterion
        ret["cls_use_bias"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_BIAS
        
        ret["use_cossim"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.USE_COSSIM
        ret["cossim_scale"] = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.COSSIM_SCALE
        ret["loss_weight"].update({"loss_contrast": cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT})
        return ret
    

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        # use clip text embeddings as classifier's weights
        if self.use_cossim: 
            normalized_x = F.normalize(x, p=2.0, dim=1)
            normalized_weights = F.normalize(self.cls_score.weight, p=2, dim=1)
            cos_dist = normalized_x @ normalized_weights.t()
            scores = self.cossim_scale * cos_dist
        # regular classifier
        else:  
            scores = self.cls_score(x)
        
        # box regression
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
    
    def losses(self, predictions, proposals, box_feat_con):
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
        ious = (
            cat([p.iou for p in proposals], dim=0) if len(proposals) else torch.empty(0.0)
        )
        _log_classification_stats(scores, gt_classes)

        if self.cl_head_only:
            return {
                "loss_contrast": self.constrastive_criterion(box_feat_con, gt_classes, ious)
            }

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
            "loss_contrast": self.constrastive_criterion(box_feat_con, gt_classes, ious)
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}