import numpy as np
import os
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from det3d.datasets.base import BaseDataset


def label_to_type(label):
    if label <= 1:
        return int(label) + 1
    else:
        return 4


class WaymoDataset(BaseDataset):
    def __init__(self,
                 info_path,
                 root_path,
                 nsweeps,
                 drop_frames=0,
                 sampler=None,
                 loading_pipelines=None,
                 augmentation=None,
                 prepare_label=None,
                 tasks=[],
                 evaluations=None,
                 create_database=False,
                 use_gt_sampling=True):

        super(WaymoDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database,
            use_gt_sampling=use_gt_sampling)

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        self.drop_frames = drop_frames
        assert 0 <= drop_frames <= 1
        self.tasks = tasks

    def read_file(self, path, timestamp=0):
        points = np.fromfile(os.path.join(
            self._root_path, path), dtype=np.float32).reshape(-1, 6)
        # x, y, z, intensity, (remove elongation, exclude nlz points)
        points = points[points[:, -1] == -1, :4]
        timelist = timestamp * np.ones((points.shape[0], 1)).astype(np.float32)
        return np.concatenate((points, timelist), axis=1)

    def load_pointcloud(self, res, info):
        lidar_path = 'lidar_point/' + info['token'] + '.bin'
        points = self.read_file(lidar_path)

        points_list = [points]
        if self.nsweeps > 1:
            for ii in range(min(self.nsweeps-1, len(info['sweeps']))):
                if self.drop_frames > 0 and np.random.uniform() < self.drop_frames:
                    continue
                prev_points = self.read_file('lidar_point/' + info['sweeps'][ii]['token'] + '.bin',
                                             timestamp=info['sweeps'][ii]['timestamp'])
                rel_pose = np.linalg.inv(
                    info['pose']) @ info['sweeps'][ii]['pose']
                prev_points[:, :3] = (np.concatenate((prev_points[:, :3], np.ones(
                    (prev_points.shape[0], 1))), axis=1) @ rel_pose.T)[:, :3]
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
            detections[token]["box3d_lidar"] = detections[token]["box3d_lidar"].detach(
            ).cpu().numpy()
            detections[token]["scores"] = detections[token]["scores"].detach(
            ).cpu().numpy()
            detections[token]["label_preds"] = detections[token]["label_preds"].detach(
            ).cpu().numpy()

        objects = metrics_pb2.Objects()
        for pred in detections:
            pred = detections[pred]
            pred_boxes = pred['box3d_lidar']
            pred_label = pred['label_preds']
            pred_score = pred['scores']

            for i in range(pred_boxes.shape[0]):
                det = pred_boxes[i]
                o = metrics_pb2.Object()
                o.context_name = pred['token'].split('-')[0]
                o.frame_timestamp_micros = int(pred['token'].split('-')[1])
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]
                o.object.box.CopyFrom(box)
                o.score = pred_score[i]
                o.object.type = label_to_type(pred_label[i])
                objects.objects.append(o)

        f = open(os.path.join(output_dir, 'waymo_preds.bin'), 'wb')
        f.write(objects.SerializeToString())
        f.close()

        print("use waymo devkit tool for evaluation")

        return {}
