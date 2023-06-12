from trainer.trainer import Trainer
from det3d.datasets.loader.build_loader import build_dataloader
from torch.nn.parallel import DistributedDataParallel
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging
import os

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path='../configs/experiments')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # distributed training
    distributed = False
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",
                                             world_size=world_size, rank=global_rank)

    # init logger before other steps
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    if distributed and local_rank != 0:
        logger.setLevel("ERROR")
    logger.info("Distributed training: {}".format(distributed))
    logger.info(
        f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

   
    val_dataset = instantiate(cfg.data.val_dataset)
    val_dataloader = build_dataloader(val_dataset, **cfg.dataloader.val)

    # build model
    model = instantiate(cfg.model)
    if distributed:
        if cfg.model.sync_batchnorm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DistributedDataParallel(
            model.cuda(local_rank), device_ids=[local_rank],
            output_device=local_rank, find_unused_parameters=False,)
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, val_dataloader=val_dataloader, logger=logger, **cfg.trainer)

    trainer.load_checkpoint(cfg.load_from, strict=True)
    trainer.val_epoch()


if __name__ == "__main__":
    main()
