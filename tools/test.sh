#!/usr/bin/bash
cd /home/jinyu/code/pillarnext/
export PATH=/opt/conda/bin:$PATH
#pip install -e .
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:/home/jinyu/code/pillarnext/det3d
export PYTHONPATH=$PYTHONPATH:/home/jinyu/code/pillarnext/trainer
umask  0000
EXP_DIR=outputs/nusc_det_pp18_aspp_iou_sp
torchrun --standalone --nnodes=1 --nproc_per_node=8  ./tools/test.py  \
		 --config-name nusc_det_pp18_aspp_iou_sp   \
	 	hydra.run.dir=outputs/$EXP_DIR  +load_from=/home/jinyu/code/pillarnext/outputs/nusc_det_pp18_aspp_iou_sp/epoch_20.pth