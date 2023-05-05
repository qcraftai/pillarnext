
import pickle
from copy import deepcopy
from typing import OrderedDict
import numpy as np
import os
from pathlib import Path

from det3d.datasets.base import BaseDataset



class WaymoDataset(BaseDataset):
    def __init__(self,
                info_path,
                root_path,
                nsweeps,
                drop_frames = 0,
                sampler=None,
                loading_pipelines = None,
                augmentation = None,
                prepare_label = None,
                tasks = [],
                evaluations=None,
                create_database=False,):

        super(WaymoDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database)

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        self.drop_frames = drop_frames
        assert  0 <= drop_frames <= 1
        self.tasks = tasks
    
    def read_file(self, path, timestamp=0):
        points = np.fromfile(os.path.join(self._root_path, path), dtype=np.float32).reshape(-1, 6)
        points = points[points[:, -1] == -1, :4] # x, y, z, intensity, (remove elongation, exclude nlz points)
        timelist =  timestamp * np.ones((points.shape[0], 1)).astype(np.float32)
        return np.concatenate((points, timelist), axis=1)

    def load_pointcloud(self, res, info):
        lidar_path = 'lidar_point/' + info['token'] +'.bin'
        points = self.read_file(lidar_path)
        
        points_list = [points]
        if self.nsweeps > 1:
            for ii in range(min(self.nsweeps-1, len(info['sweeps']))):
                if self.drop_frames > 0 and np.random.uniform() < self.drop_frames:
                    continue
                prev_points = self.read_file('lidar_point/' + info['sweeps'][ii]['token'] +'.bin',
                                            timestamp = info['sweeps'][ii]['timestamp'])
                rel_pose = np.linalg.inv(info['pose']) @ info['sweeps'][ii]['pose']
                prev_points[:, :3] = (np.concatenate((prev_points[:, :3], np.ones((prev_points.shape[0], 1))), axis=1) @ rel_pose.T)[:, :3]
                points_list.append(prev_points)

        res["points"] = np.concatenate(points_list, axis=0).astype(np.float32)
        return res

    def load_box3d(self, res, info):
        
        annos = info['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        mask_not_zero = (num_points_in_gt > 0).reshape(-1)

        gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
        gt_names = np.array([ann['label'] for ann in annos])
        
        gt_boxes = gt_boxes[mask_not_zero, :]
        gt_names = gt_names[mask_not_zero]

        res["annotations"] = {
            "gt_boxes": gt_boxes.astype(np.float32).copy(),
            "gt_names": gt_names.copy()}

        return res

    def evaluation(self, detections, output_dir=None):
        for token in detections:
            detections[token]["box3d_lidar"] = detections[token]["box3d_lidar"].detach().cpu().numpy()
            detections[token]["scores"] = detections[token]["scores"].detach().cpu().numpy()
            detections[token]["label_preds"] = detections[token]["label_preds"].detach().cpu().numpy()
        
        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".pkl"))
        with open(res_path, "wb") as f:
            pickle.dump(detections, f)

        return {}

    def eval_det3d(self, detections, output_dir=None):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)
        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None


    def eval_3Dseg(self, segmentations):
        from evaluation.segmentation.iou import ConfusionMatrix
        from tqdm import tqdm
        global_cm = ConfusionMatrix(len(self.tasks), None)
        for _, seg in tqdm(segmentations.items()):
            global_cm.update(seg['gt_seg'], seg['pred_seg'])
        results = OrderedDict()
        results['mIOU'] = global_cm.get_mean_iou()
        class_iou = global_cm.get_per_class_iou()
        for class_id, class_name in enumerate(self.tasks):
            results[class_name] = class_iou[class_id]
        
        return results
