from collections import OrderedDict

import torch
from det3d.core.bbox import box_torch_ops

import copy
from torch import nn
from det3d.models.loss.centerloss import FastFocalLoss, RegLoss, IouLoss, IouRegLoss
from det3d.models.utils.conv import ConvBlock


class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        stride=1,
        head_conv=64,
        final_kernel=1,
        bn=True,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)
        if stride > 1:
            self.deblock = ConvBlock(in_channels, head_conv, kernel_size=int(stride), 
                            stride=int(stride), padding=0, conv_layer=nn.ConvTranspose2d)
            in_channels = head_conv
        else:
            self.deblock = nn.Identity()
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = nn.Sequential()
            for i in range(num_conv-1):
                fc.append(nn.Conv2d(in_channels, head_conv,
                                    kernel_size=final_kernel, stride=1,
                                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                fc.append(nn.ReLU())

            fc.append(nn.Conv2d(head_conv, classes,
                                kernel_size=final_kernel,  stride=1,
                                padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.deblock(x)
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(
            self,
            in_channels,
            tasks,
            weight,
            code_weights,
            common_heads,
            strides,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            with_reg_iou=False,
            voxel_size=None,
            pc_range=None,
            out_size_factor=None,
            rectifier=[[0.], [0.], [0.]]):

        super(CenterHead, self).__init__()

        num_classes = [len(t) for t in tasks]
        self.class_names = tasks
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.with_reg_iou = with_reg_iou
        if self.with_reg_iou:
            self.crit_iou_reg = IouRegLoss()

        self.with_iou = 'iou' in common_heads
        if self.with_iou:
            self.crit_iou = IouLoss()

        if self.with_iou or with_reg_iou:
            self.voxel_size = voxel_size
            self.pc_range = pc_range
            self.out_size_factor = out_size_factor

        self.strides = strides

        self.rectifier = rectifier
        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        for (num_cls, stride) in zip(num_classes, strides):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=3)
            )

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind']
                                [task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads

            preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                preds_dict['vel'], preds_dict['rot']), dim=1)

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(
                preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret = OrderedDict()
            ret.update({'task': self.class_names[task_id],
                        'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss.detach().cpu(),
                        'loc_loss_elem': box_loss.detach().cpu(),
                        'num_positive': example['mask'][task_id].float().sum().cpu()})

            if self.with_iou or self.with_reg_iou:
                batch_dim = torch.exp(torch.clamp(
                    preds_dict['dim'], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict['rot'].clone()
                batch_rot = batch_rot.permute(0, 2, 3, 1).contiguous()
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)
                batch_reg = preds_dict['reg'].clone().permute(
                    0, 2, 3, 1).contiguous()
                batch_hei = preds_dict['height'].clone().permute(
                    0, 2, 3, 1).contiguous()

                batch, H, W, _ = batch_dim.size()

                batch_reg = batch_reg.reshape(batch, H * W, 2)
                batch_hei = batch_hei.reshape(batch, H * W, 1)

                batch_rot = batch_rot.reshape(batch, H * W, 1)
                batch_dim = batch_dim.reshape(batch, H * W, 3)

                ys, xs = torch.meshgrid(
                    [torch.arange(0, H), torch.arange(0, W)])
                ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

                xs = xs * self.out_size_factor[task_id] * \
                    self.voxel_size[0] + self.pc_range[0]
                ys = ys * self.out_size_factor[task_id] * \
                    self.voxel_size[1] + self.pc_range[1]

                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
                batch_box_preds = batch_box_preds.permute(
                    0, 2, 1).contiguous().reshape(batch, -1, H, W)

                if self.with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(preds_dict['iou'], example['mask'][task_id], example['ind'][task_id],
                                             pred_boxes_for_iou, example['gt_boxes'][task_id])
                    loss = loss + iou_loss
                    ret.update({'iou_loss': iou_loss.detach().cpu()})

                if self.with_reg_iou:
                    iou_reg_loss = self.crit_iou_reg(batch_box_preds, example['mask'][task_id], example['ind'][task_id],
                                                     example['gt_boxes'][task_id])
                    loss = loss + self.weight * iou_reg_loss
                    ret.update({'iou_reg_loss': iou_reg_loss.detach().cpu()})

            rets.append(ret)
            if task_id == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        return total_loss, rets

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg):
        """decode, nms, then return the detection result. Additionaly support double flip testing
        """
        # get loss info
        rets = []
        metas = []

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if "token" not in example or len(example["token"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["token"]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            if 'iou' in preds_dict.keys():
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
            else:
                batch_iou = torch.ones((batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                                       dtype=batch_dim.dtype).to(batch_hm.device)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(
                batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, H, W).repeat(
                batch, 1, 1).to(batch_hm.device).float()

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor[task_id] * \
                test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor[task_id] * \
                test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            batch_vel = preds_dict['vel']

            batch_vel = batch_vel.reshape(batch, H*W, 2)
            batch_box_preds = torch.cat(
                [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)

            metas.append(meta_list)

            rets.append(self.post_processing(task_id, batch_box_preds,
                        batch_hm, test_cfg, post_center_range, batch_iou))

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['token'] = metas[0][i]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def post_processing(self, task_id, batch_box_preds, batch_hm, test_cfg, post_center_range, batch_iou):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            iou_preds = batch_iou[i].view(-1)
            scores, labels = torch.max(hm_preds, dim=-1)
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_preds = torch.clamp(iou_preds[mask], min=0., max=1.)
            rectifier = torch.tensor(self.rectifier[task_id]).to(hm_preds)
            scores = torch.pow(
                scores, 1-rectifier[labels]) * torch.pow(iou_preds, rectifier[labels])
            selected_boxes = torch.zeros((0, 9)).to(box_preds)
            selected_labels = torch.zeros((0,), dtype=torch.int64).to(labels)
            selected_scores = torch.zeros((0,)).to(scores)
            for class_id in range(hm_preds.shape[-1]):
                scores_class = scores[labels == class_id]
                labels_class = labels[labels == class_id]
                box_preds_class = box_preds[labels == class_id]
                boxes_for_nms_class = box_preds_class[:, [
                    0, 1, 2, 3, 4, 5, -1]]
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms_class, scores_class,
                                                          thresh=test_cfg.nms.nms_iou_threshold[task_id][class_id],
                                                          pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                                          post_max_size=test_cfg.nms.nms_post_max_size)

                selected_boxes = torch.cat(
                    (selected_boxes, box_preds_class[selected]), dim=0)
                selected_scores = torch.cat(
                    (selected_scores, scores_class[selected]), dim=0)
                selected_labels = torch.cat(
                    (selected_labels, labels_class[selected]), dim=0)

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts
