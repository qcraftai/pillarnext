#!/usr/bin/bash
cd /mnt/vepfs/ML/ml-users/jinyu/offboardperception
export PATH=/opt/conda/bin:$PATH
#pip install -e .
export PYTHONPATH=$PYTHONPATH:/mnt/vepfs/ML/ml-users/jinyu/offboardperception/det3d
export PYTHONPATH=$PYTHONPATH:/mnt/vepfs/ML/ml-users/jinyu/offboardperception/trainer
export HYDRA_FULL_ERROR=1
umask  0000
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
       	./tools/train.py --config-name waymo_det_pp18_aspp_iou_car_sp \
         dataloader.train.batch_size=1  scheduler.max_lr=0.001  trainer.max_epochs=36 \
         trainer.eval_every_nepochs=36 \
	      hydra.run.dir=outputs/waymo_det_pp18_aspp_iou_car_sp \
        # +resume_from=epoch_28.pth