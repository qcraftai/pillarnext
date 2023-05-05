# PillarNeXt: Rethinking Network Designs for 3D Object Detection in LiDAR Point Clouds
## Docker
pull docker image for waymo  
`docker pull jinyuli1999/det3d:waymo`  
pull docker image for nuscenes  
`docker pull jinyuli1999/det3d:nusc`
## Usage
compile cuda extension  
`cd /your/path/pillarnext/`  
`pip install -e .`
### Prepare Data
#### ***prepare data for waymo***
```
# For Waymo Dataset         
└── WAYMO_DATASET_ROOT
    └── tfrecord
       ├── train       
       ├── val   
       ├── test
```
`python det3d/datasets/waymo/scripts/waymo_convert.py`  
create database for copy and paste  
`python det3d/datasets/utils/create_gt_database.py`
#### ***prepare data for nuscenes***
`python det3d/datasets/nuscenes/scripts/nusc_common.py`  
create database for copy and paste  
`python det3d/datasets/utils/create_gt_database.py`
### Training
`bash tools/train.sh`  
You could modify args in `train.sh`  
***Faded Stratedy***  
Do not use copy and paste stratedy in the last two epochs.
### Eval for Waymo
Use scripts to generate `*.bin` for official evaluation tools  
`python det3d/datasets/waymo/waymo_common.py`  
[waymo official evaluation tools](https://github.com/waymo-research/waymo-open-dataset/blob/r1.3/docs/quick_start.md) 
## Checkpoints
Use trained checkpoints to recurrent our results on nuScenes. [Weights for nuScenes](https://drive.google.com/file/d/1lj2q85r44_Sa-wj_nKCPx-gHDLJTA2p-/view?usp=sharing)  
`bash tools/test.sh`


