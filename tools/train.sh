#!/usr/bin/bash
cd /mnt/vepfs/ML/ml-users/jinyu/pillarnext/
export PATH=/opt/conda/bin:$PATH
#pip install -e .
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:/mnt/vepfs/ML/ml-users/jinyu/pillarnext/det3d
export PYTHONPATH=$PYTHONPATH:/mnt/vepfs/ML/ml-users/jinyu/pillarnext/trainer
# CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nnodes=1 --nproc_per_node=1   \
         ./tools/train.py --config-name waymo_det_pp18_aspp_iou_car_sp \
         dataloader.train.batch_size=3 dataloader.train.num_workers=1 scheduler.max_lr=0.0015 trainer.max_epochs=36 \
         trainer.eval_every_nepochs=36  \
         hydra.run.dir=outputs/waymo_det_pp18_aspp_iou_car_sp \
         +resume_from=/mnt/vepfs/ML/ml-users/jinyu/offboardperception/outputs/waymo_det_pp18_aspp_iou_car_sp/sota.pth

        
