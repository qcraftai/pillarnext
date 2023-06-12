
import torch
import torch.nn as nn
from torch.nn import functional as F
from det3d.core.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu


class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        '''
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.pow(out, 2) * gt * torch.log(1 - out)
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 -
                                                   pos_pred, 2) * mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return - (pos_loss + neg_loss) / num_pos


class RegLoss(nn.Module):
    '''
    Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)
        target[torch.isnan(target)] = pred[torch.isnan(
            target)].clone().detach()
        loss = F.l1_loss(pred*mask, target*mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss


class IouLoss(nn.Module):
    '''IouLoss loss for an output tensor
            Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(IouLoss, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt):
        if mask.sum() == 0:
            return iou_pred.sum() * 0
        mask = mask.bool()
        pred = _transpose_and_gather_feat(iou_pred, ind)[mask]
        pred_box = _transpose_and_gather_feat(box_pred, ind)
        target = boxes_aligned_iou3d_gpu(pred_box[mask], box_gt[mask])
        target = 2 * target - 1

        loss = F.l1_loss(pred, target, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IouRegLoss(nn.Module):
    '''Distance IoU loss for output boxes
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(IouRegLoss, self).__init__()
        self.bbox3d_iou_func = bbox3d_overlaps_diou

    def forward(self, box_pred, mask, ind, box_gt):
        if mask.sum() == 0:
            return box_pred.sum() * 0
        mask = mask.bool()
        pred_box = _transpose_and_gather_feat(box_pred, ind)
        iou = self.bbox3d_iou_func(pred_box[mask], box_gt[mask])
        loss = (1. - iou).sum() / (mask.sum() + 1e-4)
        return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
        torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5],
                      gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
        torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5],
                      pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious
