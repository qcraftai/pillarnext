import abc
import copy
import pathlib
import pickle

import numpy as np
from det3d.core.bbox import box_np_ops


class BatchSampler:
    def __init__(
        self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx: self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]
        # return np.random.choice(self._sampled_list, num)


class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass


class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict, logger=None):
        self._min_gt_point_dict = min_gt_point_dict

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


class DataBaseSamplerV2:
    def __init__(
            self,
            root_path,
            dbinfo_path,
            groups,
            db_prepor,
            rate,
            gt_drop_percentage,
            gt_drop_max_keep_points,
            point_dim):

        self.root_path = root_path
        with open(pathlib.Path(root_path) / dbinfo_path, "rb") as f:
            db_infos = pickle.load(f)

        if db_prepor is not None:
            for prepor in db_prepor.values():
                db_infos = prepor(db_infos)

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._point_dim = point_dim

        self._group_db_infos = self.db_infos  # just use db_infos
        for group_info in groups:
            group_names = list(group_info.keys())
            self._sample_classes += group_names
            self._sample_max_nums += list(group_info.values())

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k)

        self._gt_drop_rate = gt_drop_percentage
        self._gt_drop_max_keep = gt_drop_max_keep_points

    def sample_all(
        self,
        gt_boxes,
        gt_names,
    ):
        root_path = self.root_path
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes

        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(
                    class_name, sampled_num, avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            num_sampled = len(sampled)
            s_points_list = []
            for info in sampled:
                s_points = np.fromfile(
                    str(pathlib.Path(root_path) / info["path"]), dtype=np.float32
                ).reshape(-1, self._point_dim)
                #s_points = s_points[s_points[:, -1] <0.02, :]
                if "rot_transform" in info:
                    rot = info["rot_transform"]
                    s_points[:, :3] = box_np_ops.yaw_rotation(
                        s_points[:, :4], rot, axis=2
                    )
                s_points[:, :3] += info["box3d_lidar"][:3]

                s_points_list.append(s_points)

            if 1 > self._gt_drop_rate > 0:
                counts = np.zeros((num_sampled,))
                for i in range(len(s_points_list)):
                    mask = np.random.uniform(size=len(s_points_list[i]))
                    mask = mask >= self._gt_drop_rate
                    s_points_list[i] = s_points_list[i][mask]
                    counts[i] = len(s_points_list[i])
                mask2keep = counts >= self._gt_drop_max_keep
            else:
                mask2keep = np.ones((num_sampled,), dtype=np.bool_)
            # sampled_gt_boxes[:, 6:8] = 0 # np.nan
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes.astype(np.float32),
                "points": np.concatenate(s_points_list, axis=0).astype(np.float32),
                "gt_masks": mask2keep,
            }

            ret["group_ids"] = np.arange(
                gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))
        else:
            ret = None
        return ret

    def sample(self, name, num):
        ret = self._sampler_dict[name].sample(num)
        return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, [0, 1, 3, 4, -1]])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, [0, 1, 3, 4, -1]]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_np_ops.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
