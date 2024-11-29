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
    def __init__(self, input_dim, hidden_dims=512 , latent_dim=512):
        super(Encoder, self).__init__()
        # modules = []
        # last_dim = input_dim
        # for h_dim in hidden_dims:
        #     modules.append(nn.Linear(last_dim, h_dim))
        #     modules.append(nn.ReLU())  
        #     last_dim = h_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),)
        self.fc_mu = nn.Linear(hidden_dims, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=256):
        super(Decoder, self).__init__()
        # modules = []
        # last_dim = latent_dim
        # for h_dim in reversed(hidden_dims):
        #     modules.append(nn.Linear(last_dim, h_dim))
        #     modules.append(nn.ReLU())
        #     last_dim = h_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),)
        self.output_layer = nn.Linear(hidden_dims, output_dim)

    def forward(self, z):
        h = self.decoder(z)
        output = self.output_layer(h)
        output = torch.tanh(output)
        return output

class FeatureMapEncoder(nn.Module):
    def __init__(self, input_channels=1024, embed_dim=256, num_heads=8, num_layers=2, use_tf=False):
        super(FeatureMapEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.use_tf = use_tf
        if self.use_tf:
            self.proj = nn.Linear(input_channels, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, embed_dim)
            self.positional_encoding = PositionalEncoding(embed_dim)
        else:
            self.fc = nn.Linear(input_channels, embed_dim)

    def forward(self, x):
        # x: [BS, 1024, H, W]
        BS, C, H, W = x.size()
        x = x.view(BS, C, H * W)  # [BS, 1024, H*W]
        x = x.permute(0, 2, 1)    # [BS, H*W, 1024]
        if self.use_tf:
            x = self.proj(x)          # [BS, H*W, embed_dim]
            x = self.positional_encoding(x)
            # expect [sequence_length, batch_size, embed_dim]
            x = x.permute(1, 0, 2)    # [H*W, BS, embed_dim]
            x = self.transformer_encoder(x)  # [H*W, BS, embed_dim]
            x = x.mean(dim=0)         # [BS, embed_dim]
        else:
            x = x.mean(dim=1)         # [BS, 1024]
        x = self.fc(x)            # [BS, input_dim]

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [BS, seq_len, embed_dim]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].permute(1, 0, 2)  # [BS, seq_len, embed_dim]
        return x
    
class Learner(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, phase, 
                 recon_weight=1.0, reg_weight=0.0):
        super(Learner, self).__init__()
        
        self.hidden_dim = hidden_dim
        # self.attention_aggregator = AttentionAggregator(hidden_dim)
        self.projector = FeatureMapEncoder(input_channels=in_channels, embed_dim=hidden_dim)
        self.encoder = Encoder(hidden_dim, latent_dim)  # Encoding both x and y
        self.decoder = Decoder(latent_dim, hidden_dim)  # Assume y has same dimension as x
        self.phase = phase
        self.recon_weight = recon_weight
        self.reg_weight = reg_weight
        if phase == "base_train":
            self.scale_x, self.scale_y = 1.0, 1.0
        elif phase == "novel_train":
            self.scale_x, self.scale_y = 0.01, 0.1
        else:
            self.scale = 0.0


    def forward(self, x, y):
        mu, logvar = self.encoder(x) # [BS, LAT]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        y_pred = self.decoder(z) # [BS, HID]
        return y_pred, y, mu, logvar

    def forward_and_loss(self, x, y):
        x = scale_gradient(self.projector(x), self.scale_x)
        y = scale_gradient(self.projector(y), self.scale_y) 
        y_pred, y_target, mu, logvar = self.forward(x, y)
        recon_loss = F.mse_loss(y_pred, y_target)
        # reg_loss = torch.norm(z, p=2, dim=1).mean()
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / x.shape[0]
        loss = recon_loss * self.recon_weight + kld_loss * self.reg_weight 
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=1024, output_channels=2048, latent_dim=512):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, output_channels)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # 重参数技巧
        y_pred = self.decoder(z)
        return y_pred, mu, logvar
    
    def loss_function(self, y_pred, y_true, mu, logvar):
        recon_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        return loss, recon_loss, kld_loss

class Encoder(nn.Module):
    def __init__(self, input_channels=1024, latent_dim=512):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, stride=2, padding=1)  # 输出：[BS, 512, 4, 4]
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)             # 输出：[BS, 256, 2, 2]
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
    
    def forward(self, x):
        # x: [BS, 1024, 7, 7]
        h = self.relu(self.bn1(self.conv1(x)))  # [BS, 512, 4, 4]
        h = self.relu(self.bn2(self.conv2(h)))  # [BS, 256, 2, 2]
        h = self.flatten(h)                     # [BS, 256 * 2 * 2]
        mu = self.fc_mu(h)                      # [BS, latent_dim]
        logvar = self.fc_logvar(h)              # [BS, latent_dim]
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=512, output_channels=2048):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.conv_trans1 = nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1)  # 输出：[BS, 512, 4, 4]
        self.bn1 = nn.BatchNorm2d(512)
        self.conv_trans2 = nn.ConvTranspose2d(512, output_channels, kernel_size=4, stride=2, padding=1)  # 输出：[BS, 2048, 8, 8]
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        # [BS, 2048, 4, 4]
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):
        h = self.fc(z)                          # [BS, 256 * 2 * 2]
        h = h.view(h.size(0), 256, 2, 2)        # [BS, 256, 2, 2]
        h = self.relu(self.bn1(self.conv_trans1(h)))  # [BS, 512, 4, 4]
        h = self.relu(self.bn2(self.conv_trans2(h)))  # [BS, 2048, 8, 8]
        h = F.interpolate(h, size=(4, 4), mode='bilinear', align_corners=False)  # 调整到 [BS, 2048, 4, 4]
        h = self.final_conv(h)                  # [BS, 2048, 4, 4]
        return h


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

def scale_gradient(x, _lambda):
    return GradientScaling.apply(x, _lambda)

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
