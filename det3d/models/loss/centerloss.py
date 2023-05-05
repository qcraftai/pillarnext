
import torch
import torch.nn as nn
from torch.nn import functional as F
from det3d.core.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu
from det3d.core.utils.center_utils import bbox3d_overlaps_diou

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
        
        pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask.unsqueeze(2)
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
        target[torch.isnan(target)] = pred[torch.isnan(target)].clone().detach()
        loss = F.l1_loss(pred*mask, target*mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
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
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
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
