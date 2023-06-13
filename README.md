# PillarNeXt: Rethinking Network Designs for 3D Object Detection in LiDAR Point Clouds


[Jinyu Li](), [Chenxu Luo](https://chenxuluo.github.io/), [Xiaodong Yang](https://xiaodongyang.org/) <br>
PillarNeXt: Rethinking Network Designs for 3D Object Detection in LiDAR Point Clouds, CVPR 2023 <br>
[[Paper]](https://arxiv.org/pdf/2305.04925.pdf) [[Poster]]() 

<p align="left"> 
 <img src='docs/teaser_figure.png' height="410px"/> 
</p>

## Get Started

### Installation
Please refer to [INSTALL](docs/INSTALL.md) to set up environment and install dependencies. Please refer to the [Dockerfile](docker/Dockerfile) for detail.

### Data Preparation
Please refer to [DATA](docs/DATA.md) for detail. 

### Training and Evalution 
Please refer to [Training](docs/Training.md) for detail.


## Main Results
### nuScenes (val)
| Model |  mAP  |  NDS | checkpoint
| ------| -----| ---- | -------------|
 | PillarNeXt-B| 62.5 | 68.8	 | [[Google Drive]](https://drive.google.com/file/d/16abCgt-yhRGnYHQ7M259yGMO0IRYpZ8o/view?usp=drive_link)  &nbsp;&nbsp;[[Baidu Yunpan]](https://pan.baidu.com/s/1TRsjgN1ys5-mAxM70l4hog?pwd=7skt)(7skt)

### Waymo Open Dataset 
|Split | #frames | Veh L2 | Ped L2 | Cyc L2 | 
| ---------| ---------|---------|---------|---------|
| val | 1 | 67.8 | 69.8 | 69.6|
| val | 3| 72.4 | 75.2 | 75.7 |
| test| 3 | 75.8 | 76.0 | 70.6 |

All numbers are 3D mAPH. 


## Citation
 If you find this code useful in your research, please consider citing:
```
@inproceedings{li2023pillarnext,
  title={PillarNeXt: Rethinking Network Designs for 3D Object Detection in LiDAR Point Clouds},
  author={Li, Jinyu and Luo, Chenxu and Yang, Xiaodong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17567--17576},
  year={2023}
}
```

### Acknowledgement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.
* [det3d](https://github.com/poodarchu/Det3D)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
