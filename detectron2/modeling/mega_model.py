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

__all__ = ["MegaEnhancedRCNN"]


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionAggregator, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x, y):
        # x and y are assumed to be of shape [batch_size, feature_dim]
        dot_attn = x * y # [BS, DIM]
        attention = torch.tanh(self.fc(dot_attn))
        attention_weights = torch.softmax(attention, dim=-1)
        x_attn, y_attn = x * attention_weights, y * attention_weights
        return torch.cat((x_attn, y_attn), dim=1)

class RPNFeatureProjector(nn.Module):
    def __init__(self, input_channels, hidden_dim, pooled_size = 8):
        super(RPNFeatureProjector, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        # self.fc = nn.Linear(hidden_dim * pooled_size * pooled_size, hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1x1(x)  # [BS, HIDDIM, H, W]
        x = self.bn(x) 
        x = self.relu(x) # [BS, HIDDIM, H, W] 
        # x = self.global_avg_pool(x)  # [BS, HIDDIM, ps, ps]
        # x = torch.flatten(x, 1)  # [BS, HIDDIM * ps * ps]
        x = x.view(x.size(0), x.size(1), -1)  # [BS, HIDDIM, H * W]
        x = x.transpose(1, 2)  # [BS, H * W, HIDDIM]
        x = self.dropout(x)
        # x = self.fc(x)  # [BS, H * W, HIDDIM]
        return x

class RPNLearner(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, phase, 
                 recon_weight=1.0, reg_weight=0.0):
        super(RPNLearner, self).__init__()
        self.projector = RPNFeatureProjector(in_channels, hidden_dim) # [BS, HID]
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        # self.attention_aggregator = AttentionAggregator(hidden_dim)
        self.encoder = Encoder(hidden_dim, latent_dim)  # Encoding both x and y
        self.decoder = Decoder(latent_dim, hidden_dim)  # Assume y has same dimension as x
        self.phase = phase
        self.recon_weight = recon_weight
        self.reg_weight = reg_weight

        if phase == "base_train":
            pass
        elif phase == "novel_train":
            for param in self.parameters():
                param.requires_grad = False
            # for param in self.decoder.parameters():
            #     param.requires_grad = False
            pass
        else:
            pass

    def forward(self, x, y):
        x_projected = self.projector(x) # [BS, SEQ, HID]
        y_projected = self.projector(y) # [BS, SEQ, HID]
        q_proj = self.Q(x_projected) # [BS, SEQ,  HID]
        k_proj = self.K(y_projected) # [BS, SEQ, HID]
        v_proj = self.V(y_projected) # [BS, SEQ, HID]

        qk = torch.bmm(q_proj, k_proj.transpose(1, 2)) / np.sqrt(self.hidden_dim) # [BS, SEQ, SEQ]
        attn = torch.softmax(qk, dim=-1) # [BS, SEQ, SEQ]
        y_projected = torch.bmm(attn, v_proj) # [BS, SEQ, HID]
 
        z = self.encoder(y_projected) # [BS, LAT]
        y_pred = self.decoder(z) # [BS, HID]
        return y_pred, y_projected, z

    def forward_and_loss(self, x, y):
        y_pred, y_projected, z = self.forward(x, y)
        recon_loss = F.mse_loss(y_pred, y_projected)
        reg_loss = torch.norm(z, p=2, dim=1).mean()
        loss = recon_loss * self.recon_weight + reg_loss * self.reg_weight 
        return loss

# class TaskEncoder(nn.Module):
#     def __init__(self, backbone: Backbone, task_feature_dim: int = 256):
#         super(TaskEncoder, self).__init__()
#         self.backbone = backbone
#         self.task_feature_dim = task_feature_dim
        
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
#         self.embedding = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(self.backbone.output_shape()["res5"].channels, task_feature_dim),
#             nn.ReLU()
#         )
    
#     def forward(self, support_images: torch.Tensor):
#         """
#         Args:
#             support_images: Tensor of shape (N, C, H, W)
#         Returns:
#             task_feature: Tensor of shape (task_feature_dim,)
#         """
#         N = support_images.size(0)
#         device = support_images.device
#         with torch.no_grad():
#             features = self.backbone(support_images)["res5"]
#         embeddings = self.embedding(features)  # (N, task_feature_dim)
#         task_feature = embeddings.mean(dim=0)  # (task_feature_dim,)
#         return task_feature

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512] , latent_dim=512):
        super(Encoder, self).__init__()
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(last_dim, h_dim))
            modules.append(nn.ReLU())  
            last_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dims=[512], output_dim=256):
        super(Decoder, self).__init__()
        modules = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            modules.append(nn.Linear(last_dim, h_dim))
            modules.append(nn.ReLU())
            last_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        self.output_layer = nn.Linear(last_dim, output_dim)

    def forward(self, z):
        h = self.decoder(z)
        output = self.output_layer(h)
        output = torch.tanh(output)
        return output

class Learner(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, phase, 
                 recon_weight=1.0, reg_weight=0.0):
        super(Learner, self).__init__()
        
        self.hidden_dim = hidden_dim
        # self.attention_aggregator = AttentionAggregator(hidden_dim)
        self.encoder = Encoder(hidden_dim, latent_dim)  # Encoding both x and y
        self.decoder = Decoder(latent_dim, hidden_dim)  # Assume y has same dimension as x
        self.phase = phase
        self.recon_weight = recon_weight
        self.reg_weight = reg_weight


    def forward(self, x, y):
        mu, logvar = self.encoder(x) # [BS, LAT]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        y_pred = self.decoder(z) # [BS, HID]
        return y_pred, y, mu, logvar

    def forward_and_loss(self, x, y):
        y_pred, y_target, mu, logvar = self.forward(x, y)
        recon_loss = F.mse_loss(y_pred, y_target)
        # reg_loss = torch.norm(z, p=2, dim=1).mean()
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / x.shape[0]
        loss = recon_loss * self.recon_weight + kld_loss * self.reg_weight 
        return loss


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
