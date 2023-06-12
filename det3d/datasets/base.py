import os
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset
from det3d.core.bbox import box_np_ops


class BaseDataset(Dataset):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__getitem__`` supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(
            self,
            root_path,
            info_path,
            sampler=None,
            loading_pipelines=None,
            augmentation=None,
            prepare_label=None,
            evaluations=None,
            create_database=False,
            use_gt_sampling=True,):

        self._info_path = info_path
        self._root_path = Path(root_path)
        self.loading_pipelines = loading_pipelines
        self.augmentations = augmentation
        self.prepare_label = prepare_label
        self.evaluations = evaluations
        self.create_database = create_database
        self.use_gt_sampling = use_gt_sampling
        self.load_infos()
        if use_gt_sampling and sampler is not None:
            self.sampler = sampler()
        else:
            self.sampler = None

    def __len__(self):
        return len(self.infos)

    def load_infos(self):
        with open(os.path.join(self._root_path, self._info_path), "rb") as f:
            self.infos = pickle.load(f)

    def evaluation(self):
        """Dataset must provide a evaluation function to evaluate model."""
        # support different evaluation tasks
        raise NotImplementedError

    def load_pointcloud(self, res, info):
        raise NotImplementedError

    def load_box3d(self, res, info):
        res["annotations"] = {
            'gt_boxes': info["gt_boxes"].astype(np.float32).copy(),
            'gt_names': np.array(info["gt_names"]).reshape(-1).copy(),
        }

        return res

    def __getitem__(self, idx):

        info = self.infos[idx]
        res = {"token": info["token"]}

        if self.loading_pipelines is not None:
            for lp in self.loading_pipelines:
                res = getattr(self, lp)(res, info)
        if self.sampler is not None:
            sampled_dict = self.sampler.sample_all(
                res['annotations']['gt_boxes'],
                res["annotations"]['gt_names']
            )
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                res['annotations']["gt_names"] = np.concatenate(
                    [res['annotations']["gt_names"], sampled_gt_names], axis=0
                )
                res['annotations']["gt_boxes"] = np.concatenate(
                    [res['annotations']["gt_boxes"], sampled_gt_boxes]
                )

                # remove points in sampled gt boxes
                sampled_point_indices = box_np_ops.points_in_rbbox(
                    res['points'], sampled_gt_boxes[sampled_gt_masks])
                res['points'] = res['points'][np.logical_not(
                    sampled_point_indices.any(-1))]

                res['points'] = np.concatenate(
                    [sampled_points, res['points']], axis=0)
        if self.augmentations is not None:
            for aug in self.augmentations.values():
                res = aug(res)

        if self.prepare_label is not None:
            for _, pl in self.prepare_label.items():
                res = pl(res)

        if 'annotations' in res and (not self.create_database):
            del res['annotations']

        return res

    def format_eval(self):
        raise NotImplementedError
