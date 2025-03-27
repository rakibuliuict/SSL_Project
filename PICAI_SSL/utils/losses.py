import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np

def to_one_hot(tensor, n_classes):
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(1)
    assert tensor.size(1) == 1, "Expected input shape [N, 1, D, H, W]"
    size = list(tensor.size())
    size[1] = n_classes
    one_hot = torch.zeros(*size, device=tensor.device)
    return one_hot.scatter_(1, tensor, 1)

def get_probability(logits):
    if logits.size(1) > 1:
        pred = F.softmax(logits, dim=1)
        nclass = logits.size(1)
    else:
        pred = torch.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], dim=1)
        nclass = 2
    return pred, nclass

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            self.class_weights = nn.Parameter(torch.ones((1, nclass), dtype=torch.float32), requires_grad=False)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            assert nclass == class_weights.shape[0], "Mismatch in number of classes and weights."
            self.class_weights = nn.Parameter(class_weights, requires_grad=False)

    def forward(self, logits, target, mask=None):
        N, C = logits.size(0), logits.size(1)
        pred, _ = get_probability(logits)
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        target_one_hot = to_one_hot(target.long(), C).float()
        inter = pred * target_one_hot
        union = pred + target_one_hot
        if mask is not None:
            if mask.dim() == logits.dim() - 1:
                mask = mask.unsqueeze(1)
            inter = (inter * mask).view(N, C, -1).sum(dim=2)
            union = (union * mask).view(N, C, -1).sum(dim=2)
        else:
            inter = inter.view(N, C, -1).sum(dim=2)
            union = union.view(N, C, -1).sum(dim=2)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        return to_one_hot(input_tensor.unsqueeze(1), self.n_classes).float()

    def _dice_loss(self, score, target):
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes

class CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes):
        super(CrossEntropyLoss, self).__init__()
        self.class_num = n_classes

    def forward(self, inputs, target, mask):
        inputs = torch.softmax(inputs, dim=1)
        target = to_one_hot(target.unsqueeze(1), self.class_num).float()
        mask = to_one_hot(mask.unsqueeze(1), self.class_num).float()
        loss = 0.0
        for i in range(self.class_num):
            ce = (-target[:, i] * torch.log(inputs[:, i]) * mask[:, i]).sum()
            loss += ce / (mask[:, i].sum() + 1e-16)
        return loss / self.class_num

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    return 1 - intersection / union

class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            loss += Binary_dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    return d / (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)

class VAT3d(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)
        d = torch.rand_like(x).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
