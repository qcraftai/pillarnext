import numpy as np
from .center_utils import (draw_gaussian, gaussian_radius)


class AssignLabel(object):
    def __init__(self,
                 tasks,
                 gaussian_overlap,
                 max_objs,
                 min_radius,
                 pc_range,
                 voxel_size,
                 out_size_factor):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = np.array(out_size_factor)
        self.tasks = tasks
        self.gaussian_overlap = gaussian_overlap
        self._max_objs = max_objs
        self._min_radius = min_radius
        self.pc_range = np.array(pc_range)
        self.voxel_size = np.array(voxel_size)

    def __call__(self, res):
        max_objs = self._max_objs
        class_names_to_id = {}
        for ti in range(len(self.tasks)):
            t = self.tasks[ti]
            for ni in range(len(t)):  # task id, cat_id(in task)
                class_names_to_id[self.tasks[ti][ni]] = [ti, ni]

        # Calculate output featuremap size
        grid_size = (self.pc_range[3:] - self.pc_range[:3]
                     ) / self.voxel_size  # x,  y, z
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        gt_dict = res["annotations"]

        hms, annos, inds, masks, cats, gt_boxes = [], [], [], [], [], []

        for task_id, task in enumerate(self.tasks):
            feature_map_size = grid_size[:2] // self.out_size_factor[task_id]
            hm = np.zeros(
                (len(task), feature_map_size[1], feature_map_size[0]), dtype=np.float32)
            anno_box = np.zeros((max_objs, 10), dtype=np.float32)
            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat = np.zeros((max_objs), dtype=np.int64)
            gt_box = np.zeros((max_objs, 7), dtype=np.float32)

            hms.append(hm)
            annos.append(anno_box)
            inds.append(ind)
            masks.append(mask)
            cats.append(cat)
            gt_boxes.append(gt_box)

        task_nums = np.zeros(len(self.tasks), dtype=np.int64)
        for k in range(len(gt_dict['gt_names'])):
            obj_name = gt_dict['gt_names'][k]
            if obj_name not in class_names_to_id:
                continue

            x, y = gt_dict['gt_boxes'][k][0], gt_dict['gt_boxes'][k][1]
            size_x, size_y = gt_dict['gt_boxes'][k][3], gt_dict['gt_boxes'][k][4]

            task_id = class_names_to_id[obj_name][0]
            size_x = size_x / \
                self.voxel_size[0] / self.out_size_factor[task_id]
            size_y = size_y / \
                self.voxel_size[1] / self.out_size_factor[task_id]

            if size_x > 0 and size_y > 0:
                cls_id = class_names_to_id[obj_name][1]
                radius = gaussian_radius(
                    (size_y, size_x), min_overlap=self.gaussian_overlap)
                radius = max(self._min_radius, int(radius))

                # be really careful for the coordinate system of your box annotation.
                coor_x, coor_y = (x - self.pc_range[0]) / self.voxel_size[0] / self.out_size_factor[task_id], \
                    (y - self.pc_range[1]) / self.voxel_size[1] / \
                    self.out_size_factor[task_id]

                ct = np.array([coor_x, coor_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # throw out not in range objects to avoid out of array area when creating the heatmap
                if not (0 <= ct_int[0] < hms[task_id].shape[2] and 0 <= ct_int[1] < hms[task_id].shape[1]):
                    continue

                draw_gaussian(hms[task_id][cls_id], ct, radius, 1.0)

                new_idx = task_nums[task_id]

                x, y = ct_int[0], ct_int[1]
                cats[task_id][new_idx] = cls_id

                inds[task_id][new_idx] = y * hms[task_id].shape[2] + x
                masks[task_id][new_idx] = 1

                vx, vy = gt_dict['gt_boxes'][k][6:8]
                rot = gt_dict['gt_boxes'][k][8]

                annos[task_id][new_idx] = np.concatenate(
                    (ct - (x, y), gt_dict['gt_boxes'][k][2], np.log(gt_dict['gt_boxes'][k][3:6]),
                     np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                gt_boxes[task_id][new_idx] = np.concatenate(
                    (gt_dict['gt_boxes'][k][0:6],
                     gt_dict['gt_boxes'][k][8]), axis=None
                )

                task_nums[task_id] += 1

        res.update({'hm': hms, 'anno_box': annos, 'ind': inds,
                   'mask': masks, 'cat': cats, 'gt_boxes': gt_boxes})

        return res
