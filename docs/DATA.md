# Data Preparation
## nuScenes
Download data and organize as follows
```       
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```
### create data
```
python tools/create_data  nuscenes_data_prep --root_path /path/to/nuscenes
```

##  Waymo Open Dataset   
```      
└── WAYMO_DATASET_ROOT
    └── tfrecord
       ├── train       
       ├── val   
       ├── test
```
```
python tools/create_data.py waymo_data_prep /path/to/raw/waymo/ /path/to/save/converted_waymo/` to generate data for waymo
```