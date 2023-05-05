import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List

from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from .nusc_common import (general_to_detection, cls_attr_dist, quaternion_yaw)


def get_previous_sweeps(nusc, lidar_token, lidar_time, nsweeps, ref_from_car, car_from_global):
    from nuscenes.utils.geometry_utils import transform_matrix
    curr_lidar_rec = nusc.get("sample_data", lidar_token)
    sweeps = []
    while len(sweeps) < nsweeps - 1:
        if curr_lidar_rec["prev"] == "":
            break
        else:
            curr_lidar_rec = nusc.get("sample_data", curr_lidar_rec["prev"])

            # Get past pose
            current_pose_rec = nusc.get("ego_pose", curr_lidar_rec["ego_pose_token"])
            global_from_car = transform_matrix(
                current_pose_rec["translation"],
                Quaternion(current_pose_rec["rotation"]),
                inverse=False,
            )

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get(
                "calibrated_sensor", curr_lidar_rec["calibrated_sensor_token"]
            )
            car_from_current = transform_matrix(
                current_cs_rec["translation"],
                Quaternion(current_cs_rec["rotation"]),
                inverse=False)

            tm = reduce(np.dot,
                [ref_from_car, car_from_global, global_from_car, car_from_current])

            lidar_path = nusc.get_sample_data_path(curr_lidar_rec["token"])

            time_lag = lidar_time - 1e-6 * curr_lidar_rec["timestamp"]

            sweep = {
                "lidar_path": lidar_path,
                "sample_data_token": curr_lidar_rec["token"],
                "transform_matrix": tm,
                "global_from_car": global_from_car,
                "car_from_current": car_from_current,
                "time_lag": time_lag,
            }
            sweeps.append(sweep)
    
    return sweeps
        
def _get_sample_info(nusc, sample, test=False, nsweeps=10, max_time_diff=1.5):
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = nusc.get_sample_data_path(lidar_token)

    lidar_rec = nusc.get("sample_data", lidar_token)

    lidar_calib_rec = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
    ego_pose_rec = nusc.get("ego_pose", lidar_rec["ego_pose_token"])
    lidar_time = 1e-6 * lidar_rec["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(lidar_calib_rec["translation"], Quaternion(lidar_calib_rec["rotation"]), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ego_pose_rec["translation"],Quaternion(ego_pose_rec["rotation"]),inverse=True)

    info = {
        "lidar_path": lidar_path,
        "token": sample["token"],
        "sweeps": [],
        "ref_from_car": ref_from_car,
        "car_from_global": car_from_global,
        "timestamp": lidar_time,
    }

    gt_boxes_list = []
    prev_boxes_list = []
    names_list = []
    tokens_list = []
    
    for sample_annotation_token in sample['anns']:
        record = nusc.get('sample_annotation', sample_annotation_token)
        #if record['num_lidar_pts'] == 0
        box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

        # Move box to ego vehicle coord system
        box.translate(-np.array(ego_pose_rec["translation"]))
        box.rotate(Quaternion(ego_pose_rec["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(lidar_calib_rec["translation"]))
        box.rotate(Quaternion(lidar_calib_rec["rotation"]).inverse)

        # get velocity estimation 
        has_prev = record["prev"] != ""
        has_next = record["next"] != ""
        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            velo = np.array([np.nan, np.nan])

        else:
            if has_prev:
                first = nusc.get("sample_annotation", record["prev"])
            else:
                first = record

            if has_next:
                last = nusc.get("sample_annotation", record["next"])
            else:
                last = record

            pos_last = np.array(last["translation"])
            pos_first = np.array(first["translation"])
            pos_diff = pos_last - pos_first

            time_last = 1e-6 * nusc.get("sample", last["sample_token"])["timestamp"]
            time_first = 1e-6 * nusc.get("sample", first["sample_token"])["timestamp"]
            time_diff = time_last - time_first

            if has_next and has_prev:
                # If doing centered difference, allow for up to double the max_time_diff.
                max_time_diff *= 2

            if time_diff > max_time_diff:
                # If time_diff is too big, don't return an estimate.
                velocity = np.array([np.nan, np.nan])
                #print('exceed!')
            else:
                assert time_diff > 0
                velocity = pos_diff[:2] / time_diff
            
            velo = np.array([velocity[0], velocity[1], 0.0])
            velo = ref_from_car[:3, :3] @ car_from_global[:3, :3] @ velo
        
        # box annotations
        # format: (x,y,z), (l, w, h), vx, vy, theta(rz)
        gt_boxes_list.append(np.hstack((box.center, box.wlh[[1,0,2]], velo[:2], quaternion_yaw(box.orientation))))
        names_list.append(box.name)
        tokens_list.append(box.token)
        
        if has_prev and first['sample_token'] == sample['prev']:
            prev_box = Box(first['translation'], first['size'], Quaternion(first['rotation']),
                   name=first['category_name'], token=first['token'])

            # move previous box to current lidar coordinates
            prev_box.translate(-np.array(ego_pose_rec["translation"]))
            prev_box.rotate(Quaternion(ego_pose_rec["rotation"]).inverse)
            prev_box.translate(-np.array(lidar_calib_rec["translation"]))
            prev_box.rotate(Quaternion(lidar_calib_rec["rotation"]).inverse)
            
            prev_gt_boxes = np.hstack((prev_box.center, prev_box.wlh[[1,0,2]], quaternion_yaw(prev_box.orientation)))
            
        else:
            prev_gt_boxes = np.array([np.nan]*7)
        
        prev_boxes_list.append(prev_gt_boxes)

    if len(gt_boxes_list) > 0:
        info["gt_boxes"] = np.stack(gt_boxes_list)
        info["gt_names"] = np.array([general_to_detection[name] for name in names_list])
        info["gt_boxes_token"] = tokens_list
        info['prev_gt_boxes'] = np.stack(prev_boxes_list)
    else:
        info["gt_boxes"] = np.empty((0,9))
        info["gt_names"] = np.array([])
        info["gt_boxes_token"] = []
        info['prev_gt_boxes'] = np.empty((0,7))

    assert info['gt_boxes'].shape[0] == len(info['gt_names']) == len(tokens_list) == info['prev_gt_boxes'].shape[0]
    
    info['sweeps'] = get_previous_sweeps(nusc, lidar_token, lidar_time, nsweeps, ref_from_car, car_from_global)
    
    return info
            

def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=10):

    train_nusc_infos = []
    val_nusc_infos = []

    for sample in tqdm(nusc.sample):
        info = _get_sample_info(nusc, sample, test=test, nsweeps=nsweeps)

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_tracking_infos(root_path, version="v1.0-trainval", nsweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = [scene for scene in nusc.scene]
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc, train_scenes, val_scenes, test, nsweeps=nsweeps)

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(root_path / "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps), "wb") as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
        with open(root_path / "infos_train_{:02d}sweeps_withvelo_format_tracking.pkl".format(nsweeps), "wb") as f:
            pickle.dump(train_nusc_infos, f)
        with open(root_path / "infos_val_{:02d}sweeps_withvelo_format_tracking.pkl".format(nsweeps), "wb") as f:
            pickle.dump(val_nusc_infos, f)


def get_box_mean(info_path, class_name="vehicle.car"):
    with open(info_path, "rb") as f:
        nusc_infos = pickle.load(f)

    gt_boxes_list = []
    for info in nusc_infos:
        mask = np.array([s == class_name for s in info["gt_names"]], dtype=np.bool_)
        gt_boxes_list.append(info["gt_boxes"][mask].reshape(-1, 7))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    print(gt_boxes_list.mean(0))


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0,)
