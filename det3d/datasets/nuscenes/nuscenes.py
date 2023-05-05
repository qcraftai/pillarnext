import json
import operator
import numpy as np
import os
import itertools
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from det3d.datasets.base import BaseDataset

from det3d.datasets.nuscenes.scripts.nusc_common import (
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)

class NuScenesDataset(BaseDataset):

    def __init__(self,
                info_path,
                root_path,
                nsweeps,
                sampler=None,
                loading_pipelines = None,
                augmentation = None,
                prepare_label = None,
                class_names=[],
                resampling=False,
                evaluations=None,
                create_database=False,
                version="v1.0-trainval"):
        
        super(NuScenesDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database)

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"

        self._class_names = list(itertools.chain(*[t for t in class_names]))
        self.version = version

        if resampling:
            self.cbgs()

    def cbgs(self):
        _cls_infos = {name: [] for name in self._class_names}
        for info in self.infos:
            for name in set(info["gt_names"]):
                if name in self._class_names:
                    _cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

        _nusc_infos = []

        frac = 1.0 / len(self._class_names)
        ratios = [frac / v for v in _cls_dist.values()]

        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
            _nusc_infos += np.random.choice(cls_infos, int(len(cls_infos) * ratio)).tolist()
        
        self.infos = _nusc_infos


    def read_file(self, path, num_point_feature=4):
        points = np.fromfile(os.path.join(self._root_path, path), dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        return points

    def read_sweep(self, sweep, min_distance = 1.0):
        points_sweep = self.read_file(str(sweep["lidar_path"])).T

        nbr_points = points_sweep.shape[1]
        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points))))[:3, :]
        points_sweep = self.remove_close(points_sweep, min_distance)
        curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

        return points_sweep.T, curr_times.T
    
    @staticmethod
    def remove_close(points, radius: float):
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        return points

    def load_pointcloud(self, res, info):

        lidar_path = info["lidar_path"]
        
        points = self.read_file(str(lidar_path))

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]


        for i in range(len(info["sweeps"])):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = self.read_sweep(sweep)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        res["points"] = np.hstack([points, times])

        return res
    

    def evaluation(self, detections, output_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        dets = [v for _, v in detections.items()]
        assert len(dets) == 6019

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        mapped_class_names = []
        for n in self._class_names:
            mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                "detection_cvpr_2019",
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
            return res['results']
        else:
            return None

        