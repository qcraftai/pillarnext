# Training and Evaluation

## nuScenes
After preparing data, run the folloing script to train the model. By default, all models are trained on 8 GPUs.
```
torchrun --standalone --nnodes=1 --nproc_per_node=8   \
         ./tools/train.py --config-name nusc_det_pp18_aspp_iou_sp \
         data.train_dataset.root_path=/root/to/nuscenes/ \
         dataloader.train.batch_size=6 \
         scheduler.max_lr=0.003 \
         trainer.max_epochs=20 \
         hydra.run.dir=outputs/nusc_pillarnextb
```

For the reported results, we apply ***Faded Stratedy***, where the copy and paste are remove in the last two epochs. You can add `+data.train_dataset.use_gt_sampling=False` to disable copy and paste. Currently, we manually stopped the training at epoch 18 and restart the training with the following script:
```
torchrun --standalone --nnodes=1 --nproc_per_node=8   \
         ./tools/train.py --config-name nusc_det_pp18_aspp_iou_sp \
         data.train_dataset.root_path=/root/to/nuscenes/ \
         dataloader.train.batch_size=6 \
         scheduler.max_lr=0.003 \ 
         trainer.max_epochs=20 \
         hydra.run.dir=outputs/nusc_pillarnextb \
         +data.train_dataset.use_gt_sampling=False \
         +resume_from=epoch_18.pth
```

## Waymo Open Dataset
```
torchrun --standalone --nnodes=1 --nproc_per_node=8   \
         ./tools/train.py \
         --config-name waymo_det_pp18_aspp_iou_car_sp \
         data.train_dataset.root_path=/path/to/waymo/ \
         dataloader.train.batch_size=3 \
         scheduler.max_lr=0.0015 \
         trainer.max_epochs=36 \
         trainer.eval_every_nepochs=36  \
         hydra.run.dir=outputs/waymo_pillarnextb 
```
For Waymo, we apply faded strategy in the last 4 epochs.

Note: The results repored in Table 4-6 are trained on 32 GPUs, you can refer to this [script](tools/dist_train_waymo.sh) for detail. The performance may be slightly different if you are only using 8 GPUs. 

For evaluation, please use the official [evaluation tools](https://github.com/waymo-research/waymo-open-dataset/blob/r1.3/docs/quick_start.md) 
