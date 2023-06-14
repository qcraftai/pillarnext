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
Run
```
python tools/create_data  nuscenes_data_prep --root_path /path/to/nuscenes
```

##  Waymo Open Dataset 
Download data and organize as follows  
```      
└── WAYMO_DATASET_ROOT
    └── tfrecord
       ├── train       
       ├── val   
       ├── test
```
Run
```
python tools/create_data.py waymo_data_prep /path/to/raw/waymo/ /path/to/save/converted_waymo/ 
```
