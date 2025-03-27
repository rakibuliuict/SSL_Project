import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib


def to_one_hot(tensor, n_classes):
    """
    Convert a tensor of shape [N, 1, D, H, W] to one-hot encoding of shape [N, C, D, H, W]
    """
    assert tensor.dim() == 5, f"Expected 5D input, got {tensor.shape}"
    N, _, D, H, W = tensor.size()
    one_hot = torch.zeros((N, n_classes, D, H, W), device=tensor.device)
    return one_hot.scatter_(1, tensor.long(), 1)


def get_probability(logits):
    """
    Apply softmax or sigmoid to get prediction probabilities from raw logits.
    Assumes input logits are [N, C, D, H, W]
    """
    if logits.size(1) > 1:
        pred = F.softmax(logits, dim=1)
        nclass = logits.size(1)
    else:
        pred = torch.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], dim=1)
        nclass = 2
    return pred, nclass

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.nclass = nclass
        self.smooth = smooth

    def forward(self, logits, target, mask=None):
        """
        logits: [N, C, D, H, W]
        target: [N, D, H, W]
        mask:   [N, D, H, W] or [N, 1, D, H, W]
        """
        if logits.dim() == 4:
            # Reshape 2D input to 3D shape for consistency
            logits = logits.unsqueeze(2)  # [N, C, 1, H, W]
            target = target.unsqueeze(1)  # [N, 1, H, W] → [N, 1, 1, H, W]
        elif logits.dim() != 5:
            raise ValueError(f"Unsupported input shape: {logits.shape}")

        N, C, D, H, W = logits.shape

        pred, _ = get_probability(logits)

        if target.dim() == 4:
            target = target.unsqueeze(1)  # [N, 1, D, H, W]

        try:
            target_one_hot = to_one_hot(target, C).float()
        except Exception as e:
            print(f"[to_one_hot] error: {e}")
            print(f"target shape: {target.shape}, logits shape: {logits.shape}")
            raise e

        # print(f"[DEBUG] pred shape: {pred.shape}, target_one_hot shape: {target_one_hot.shape}")

        # Compute Dice loss
        inter = pred * target_one_hot
        union = pred + target_one_hot

        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)
            inter = (inter * mask).view(N, C, -1).sum(2)
            union = (union * mask).view(N, C, -1).sum(2)
        else:
            inter = inter.view(N, C, -1).sum(2)
            union = union.view(N, C, -1).sum(2)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss







class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


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
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


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

        d = torch.rand(x.shape).sub(0.5).to(x.device)
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
