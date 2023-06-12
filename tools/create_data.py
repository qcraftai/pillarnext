import fire
from det3d.datasets.nuscenes.nusc_common import create_nuscenes_infos 
from det3d.datasets.waymo.waymo_convert import create_waymo_infos
from create_gt_database import create_groundtruth_database


def nuscenes_data_prep(root_path, version="v1.0-trainval", nsweeps=10):
    create_nuscenes_infos(root_path)
    create_groundtruth_database('NUSC', 
                                root_path, 
                                'infos_train_10sweeps_withvelo_filterZero.pkl',
                                nsweeps=nsweeps)


def waymo_data_prep(root_path, save_path, nsweeps=3):
    create_waymo_infos(root_path, save_path)
    create_groundtruth_database('WAYMO', 
                                save_path, 
                                'waymo_infos_train.pkl',
                                nsweeps=nsweeps)


if __name__ == '__main__':
    fire.Fire()