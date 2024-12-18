import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        feature_l2 = torch.norm(features, dim=1)
        features = features / feature_l2.unsqueeze(1)
        cos_sim = torch.matmul(features, features.T) 
        similarity = cos_sim / self.temperature

        # similarity = torch.div(
        #     torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        # sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        # similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)
        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob
        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]
        loss = loss * coef

        # exp_sim = torch.exp(similarity)
        # mask = logits_mask * label_mask
        # keep = (mask.sum(1) != 0 ) & (ious >= self.iou_threshold)
        # print(exp_sim)
        # log_prob = torch.log(
        #     (exp_sim[keep] * mask[keep]).sum(1) / (exp_sim[keep] * logits_mask[keep]).sum(1)
        # )

        loss = -log_prob

        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay
        

class ContrastiveEncoder(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized