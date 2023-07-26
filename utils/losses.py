from torch import nn
from utils import measures
import torch
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(self, bce_coef=0.5, dice_coef=0.5, vertex_abs=False):

        super(SegLoss, self).__init__()
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef
        self.vertex_loss = vertex_abs
        self.mse = torch.nn.MSELoss(size_average=True)

    def forward(self, pred_seg, gt_seg):
        mask = gt_seg[:, 0, ...].clone()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        mask = gt_seg.clone()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

        dice = measures.dice_loss(pred_seg[:, :2, ...], gt_seg[:, :2, ...])
        mean_dice = torch.mean(dice)
        mean_cross_entropy = F.binary_cross_entropy(pred_seg[:, :2, ...], gt_seg[:, :2, ...], weight=mask[:, :2, ...],
                                                    reduction="mean")
        if self.vertex_loss:
            gt_seg_vertex = gt_seg[:, 2, ...]
            pred_vertex = pred_seg[:, 2, ...]
            # avg_vertex_loss = torch.mean(torch.abs(gt_seg_vertex - pred_vertex))
            mean_cross_entropy += self.mse(pred_vertex, gt_seg_vertex)
        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice


class BinSegLoss(nn.Module):
    def __init__(self, bce_coef=0.5, dice_coef=0.5):

        super(BinSegLoss, self).__init__()
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def forward(self, pred_seg, gt_seg):

        num_positive = torch.sum((gt_seg == 1).float()).float()
        num_negative = torch.sum((gt_seg == 0).float()).float()
        mask = gt_seg.clone()
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

        dice = measures.dice_loss(pred_seg, gt_seg)
        mean_dice = torch.mean(dice)
        mean_cross_entropy = F.binary_cross_entropy(pred_seg, gt_seg, weight=mask, reduction="mean")

        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice