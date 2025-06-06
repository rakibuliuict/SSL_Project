import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    # def forward(self, logits, target, mask=None):
    #     size = logits.size()
    #     N, nclass = size[0], size[1]

    #     logits = logits.view(N, nclass, -1)
    #     target = target.view(N, 1, -1)

    #     pred, nclass = get_probability(logits)

    #     # N x C x H x W
    #     pred_one_hot = pred
    #     target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

    #     # N x C x H x W
    #     inter = pred_one_hot * target_one_hot
    #     union = pred_one_hot + target_one_hot

    #     if mask is not None:
    #         mask = mask.view(N, 1, -1)
    #         inter = (inter.view(N, nclass, -1) * mask).sum(2)
    #         union = (union.view(N, nclass, -1) * mask).sum(2)
    #     else:
    #         # N x C
    #         inter = inter.view(N, nclass, -1).sum(2)
    #         union = union.view(N, nclass, -1).sum(2)

    #     # smooth to prevent overfitting
    #     # [https://github.com/pytorch/pytorch/issues/1249]
    #     # NxC
    #     dice = (2 * inter + self.smooth) / (union + self.smooth)
    #     return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        N, C = logits.size(0), logits.size(1)
        D, H, W = logits.shape[2:]

        # [N, C, D, H, W] → [N, C, D*H*W]
        logits = logits.view(N, C, -1)
        pred = F.softmax(logits, dim=1)

        # [N, 1, D, H, W] → [N, 1, D*H*W]
        if target.dim() == 5:
            target = target.view(N, 1, -1)
        elif target.dim() == 4:
            target = target.unsqueeze(1).view(N, 1, -1)
        else:
            raise ValueError("Target shape must be [N, D, H, W] or [N, 1, D, H, W]")

        target_one_hot = to_one_hot(target, C).float()

        inter = pred * target_one_hot
        union = pred + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter * mask).sum(dim=2)
            union = (union * mask).sum(dim=2)
        else:
            inter = inter.sum(dim=2)
            union = union.sum(dim=2)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_logits) ** 2
    return mse_loss


def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    DICE = DiceLoss(2)
    CE = nn.CrossEntropyLoss(reduction='none')
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss


def mix_mse_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 2)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 2)
    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight

    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss

voxel_kl_loss = nn.KLDivLoss(reduction="none")

def mix_max_kl_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    with torch.no_grad():
        s1 = torch.softmax(net3_output, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        img_diff_mask = (l1 != img_l)
        patch_diff_mask = (l1 != patch_l)

        uniform_distri = torch.ones(net3_output.shape)
        uniform_distri = uniform_distri.cuda()
    
    kl_loss = torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri), dim=1) * mask * img_diff_mask * image_weight
    kl_loss += torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri), dim=1) * patch_mask * patch_diff_mask * patch_weight

    sum_diff = torch.sum(mask * img_diff_mask * diff_mask) + torch.sum(patch_mask * patch_diff_mask * diff_mask)

    loss = torch.sum(diff_mask * kl_loss) / (sum_diff + 1e-16)
    return loss

