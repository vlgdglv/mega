# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

__all__ = ["MegaEnhancedRCNN"]


class TaskEncoder(nn.Module):
    def __init__(self, backbone: Backbone, task_feature_dim: int = 256):
        super(TaskEncoder, self).__init__()
        self.backbone = backbone
        self.task_feature_dim = task_feature_dim
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.output_shape()["res5"].channels, task_feature_dim),
            nn.ReLU()
        )
    
    def forward(self, support_images: torch.Tensor):
        """
        Args:
            support_images: Tensor of shape (N, C, H, W)
        Returns:
            task_feature: Tensor of shape (task_feature_dim,)
        """
        N = support_images.size(0)
        device = support_images.device
        with torch.no_grad():
            features = self.backbone(support_images)["res5"]
        embeddings = self.embedding(features)  # (N, task_feature_dim)
        task_feature = embeddings.mean(dim=0)  # (task_feature_dim,)
        return task_feature

class FiLM(nn.Module):
    def __init__(self, task_feature_dim: int, num_channels: int):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(task_feature_dim, num_channels)
        self.beta = nn.Linear(task_feature_dim, num_channels)
    
    def forward(self, x: torch.Tensor, task_feature: torch.Tensor):
        gamma = self.gamma(task_feature).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, C, 1, 1)
        beta = self.beta(task_feature).unsqueeze(0).unsqueeze(2).unsqueeze(3)    # (1, C, 1, 1)
        return x * gamma + beta


class GradientScaling(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None

class GradientScalingLayer(nn.Module):
    def __init__(self, num_channels, bias=True, scale=1.0):
        super(GradientScalingLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.scale = scale

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, x, scale=None):
        x = GradientScaling.apply(x, scale if scale is not None else self.scale)
        
        out = x * self.weight.expand_as(x)
        if self.bias is not None:
            out = out + self.bias.expand_as(x)
        
        return out
