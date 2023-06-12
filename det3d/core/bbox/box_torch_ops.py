import torch
from det3d.core.iou3d_nms import iou3d_nms_cuda


def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    :param boxes: (N, 7) [x, y, z, size_x, size_y, size_z, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # transform back to pcdet's coordinate
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    selected = order[keep[:num_out].cuda()].contiguous()

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected
