from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset import dataset_pb2
import zlib
import glob
import pickle
from tqdm import tqdm
import numpy as np
import os
from multiprocessing import Pool
import copy
import itertools

import tensorflow.compat.v2 as tf
import fire

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']


def extract_points(lasers, laser_calibrations, frame_pose):
    def sort_lambda(x): 
        return x.name
    lasers_with_calibration = zip(sorted(lasers, key=sort_lambda), sorted(
        laser_calibrations, key=sort_lambda))

    points_all = []
    for laser, calibration in lasers_with_calibration:
        points_list = extract_points_from_range_image(
            laser, calibration, frame_pose)
        points = np.concatenate(points_list, axis=0)
        points[..., 3] = np.tanh(points[..., 3])
        points_all.extend(points.astype(np.float32))

        return {'points': np.asarray(points_all)}


def extract_points_from_range_image(laser, calibration, frame_pose):

    if laser.name != calibration.name:
        raise ValueError('Laser and calibration do not match')

    if laser.name == dataset_pb2.LaserName.TOP:
        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame_pose.transform), [4, 4]))
        range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
            zlib.decompress(laser.ri_return1.range_image_pose_compressed))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(
            range_image_top_pose.data), range_image_top_pose.shape.dims)

        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[..., 2])

        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]

        range_image_top_pose_tensor = transform_utils.get_transform(range_image_top_pose_tensor_rotation,
                                                                    range_image_top_pose_tensor_translation)

        frame_pose = tf.expand_dims(frame_pose, axis=0)
        pixel_pose = tf.expand_dims(range_image_top_pose_tensor, axis=0)
    else:
        pixel_pose = None
        frame_pose = None

    first_return = zlib.decompress(laser.ri_return1.range_image_compressed)
    second_return = zlib.decompress(laser.ri_return2.range_image_compressed)

    points_list = []

    for range_image_str in [first_return, second_return]:
        range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
        if not calibration.beam_inclinations:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([calibration.beam_inclination_min,
                            calibration.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(calibration.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(
            np.array(calibration.extrinsic.transform), [4, 4])
        range_image_tensor = tf.reshape(tf.convert_to_tensor(
            range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = (
            range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(
                    beam_inclinations), axis=0),
                pixel_pose=pixel_pose,
                frame_pose=frame_pose))

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(tf.concat([range_image_cartesian, range_image_tensor[..., 1:4]], axis=-1),
                                     tf.where(range_image_mask)).numpy()

        # x, y, z, intensity, elongation, nlz (1 for nlz)
        points_tensor = points_tensor[:, :6]

        points_list.append(points_tensor)

    return points_list


def extract_objects(laser_labels, pose):
    objects = []
    for object_id, label in enumerate(laser_labels):
        category_label = label.type
        box = label.box

        num_lidar_points_in_box = label.num_lidar_points_in_box
        speed = np.array(
            [label.metadata.speed_x, label.metadata.speed_y]).reshape([2, 1])

        speed = pose[:2, :2].T @ speed

        objects.append({
            'id': label.id,
            'label': TYPE_LIST[category_label],
            'box': np.array([box.center_x, box.center_y, box.center_z,
                            box.length, box.width, box.height,
                            speed[0][0], speed[1][0],
                            box.heading],
                            dtype=np.float32),
            'num_points': num_lidar_points_in_box})

    return objects


def extract_images(frame):
    images = sorted(frame.images, key=lambda i: i.name)
    images_list = {}
    for cam_idx in range(len(images)):
        img = tf.image.decode_jpeg(images[cam_idx].image).numpy()
        images_list[images[cam_idx].name] = copy.deepcopy(img)
    return images_list


def convert(scene, save_root):
    fname = scene
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    all_frames = []
    for frame_id, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # lidar
        lidars = extract_points(
            frame.lasers, frame.context.laser_calibrations, frame.pose)
        # 3D annotation
        pose = np.reshape(np.array(frame.pose.transform), [4, 4])
        objects = extract_objects(frame.laser_labels, pose)

        # frame pose

        token = '-'.join((frame.context.name, str(frame.timestamp_micros)))

        points = lidars['points'].reshape(-1).astype(np.float32)
        lidar_path = os.path.join('lidar_point', token+'.bin')
        
        points.tofile(os.path.join(save_root, lidar_path))

        sweeps = []
        for ii in range(1, 5):
            sw = {}
            prev_frame_id = frame_id - ii
            if prev_frame_id < 0:
                break

            sw['pose'] = all_frames[prev_frame_id]['pose']
            sw['token'] = all_frames[prev_frame_id]['token']
            sw['timestamp'] = (frame.timestamp_micros -
                               int(sw['token'].split('-')[1])) / 1e6
            sweeps.append(sw)

        info = {
            'token': token,
            'pose': pose,
            'sweeps': sweeps,
            'objects': objects,
        }

        all_frames.append(copy.deepcopy(info))
    return all_frames

def create_waymo_infos(root_path, save_path):
    os.makedirs(os.path.join(save_path, 'lidar_point'), exist_ok=True)
    for split in ['train','val']:
        all_infos = []
        scenes = list(glob.glob(os.path.join(root_path, split, '*.tfrecord')))
        for scene in tqdm(scenes):
            all_infos += convert(scene, save_path)
        with open(os.path.join(save_path, 'waymo_infos_' + split + '.pkl'), 'wb') as f:
            pickle.dump(all_infos, f)


if __name__ == '__main__':
    fire.Fire()
