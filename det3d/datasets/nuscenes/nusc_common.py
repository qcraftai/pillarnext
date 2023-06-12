import numpy as np
import pickle

from pathlib import Path
from functools import reduce

from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
import fire

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def box_velocity(nusc, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to np.nan.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
    """

    current = nusc.get("sample_annotation", sample_annotation_token)
    has_prev = current["prev"] != ""
    has_next = current["next"] != ""

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.array([np.nan, np.nan, np.nan])

    if has_prev:
        first = nusc.get("sample_annotation", current["prev"])
    else:
        first = current

    if has_next:
        last = nusc.get("sample_annotation", current["next"])
    else:
        last = current

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
        return np.array([np.nan, np.nan, np.nan])
    else:
        return pos_diff / time_diff


def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d = box3d[:, [0, 1, 2, 4, 3, 5, 6, 7, 8]]
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_boxes(nusc, sample_data_token, selected_anntokens = None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:

        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return box_list


def _fill_trainval_infos(nusc, train_scenes, val_scenes, nsweeps=10, **kwargs):
    from nuscenes.utils.geometry_utils import transform_matrix

    train_nusc_infos = []
    val_nusc_infos = []

    for sample in tqdm(nusc.sample):
        """ Manual save info["sweeps"] """
        # Get reference pose and timestamp
        ref_sd_token = sample["data"]["LIDAR_TOP"]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get("calibrated_sensor",
                              ref_sd_rec["calibrated_sensor_token"])
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_boxes = get_boxes(nusc, ref_sd_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        info = {
            "lidar_path": ref_sd_rec['filename'],
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }

        sample_data_token = sample["data"]["LIDAR_TOP"]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec["prev"] == "":
                break
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # Get past pose
                current_pose_rec = nusc.get(
                    "ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False)

                tm = reduce(
                    np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": curr_sd_rec['filename'],
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        if sample["scene_token"] in train_scenes:
            annotations = [nusc.get("sample_annotation", token)
                           for token in sample["anns"]]
            mask = np.array(
                [(anno['num_lidar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
            dims = dims[:, [1, 0, 2]]   # convert to waymo box format

            velocity = np.array([box_velocity(nusc, token)[:2]
                                for token in sample['anns']])
            for i in range(len(ref_boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = ref_from_car[:3, :3] @ car_from_global[:3, :3] @ velo
                velocity[i] = velo[:2]
            velocity = velocity.reshape(-1, 2)

            rots = np.array([quaternion_yaw(b.orientation)
                            for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            gt_boxes = np.concatenate(
                [locs, dims, velocity[:, :2], rots], axis=1)

            assert len(annotations) == len(gt_boxes)

            info["gt_boxes"] = gt_boxes[mask]
            info["gt_names"] = np.array(
                [general_to_detection[name] for name in names])[mask]

            train_nusc_infos.append(info)

        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    # Measure yaw using arctan.
    return np.arctan2(v[1], v[0])


def create_nuscenes_infos(root_path, version="v1.0-trainval", nsweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = []
        val_scenes = splits.test
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes])
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes])

    print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, nsweeps)

    if test:
        print(f"test sample: {len(val_nusc_infos)}")
        with open(root_path / "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps), "wb") as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
        with open(root_path / "infos_train_{:02d}sweeps_withvelo_filterZero.pkl".format(nsweeps), "wb") as f:
            pickle.dump(train_nusc_infos, f)
        with open(root_path / "infos_val_{:02d}sweeps_withvelo_filterZero.pkl".format(nsweeps), "wb") as f:
            pickle.dump(val_nusc_infos, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    _ = nusc_eval.main(plot_examples=0,)


if __name__ == '__main__':
    # create_nuscenes_infos
    fire.Fire()