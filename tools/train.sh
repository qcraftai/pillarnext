#!/usr/bin/bash
cd /home/jinyu/code/pillarnext
export PATH=/opt/conda/bin:$PATH
#pip install -e .
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:/home/jinyu/code/pillarnext/det3d
export PYTHONPATH=$PYTHONPATH:/home/jinyu/code/pillarnext/trainer
# CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nnodes=1 --nproc_per_node=1   \
         ./tools/train.py --config-name waymo_det_pp18_aspp_iou_car_sp \
         dataloader.train.batch_size=4  scheduler.max_lr=0.002 trainer.max_epochs=12 \
         trainer.eval_every_nepochs=12  \
         hydra.run.dir=outputs/waymo_det_pp18_aspp_iou_car_sp \
        #  +resume_from=epoch_12.pth

        
