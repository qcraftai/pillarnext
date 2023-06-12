import torch
from pathlib import Path
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.progressbar import ProgressBar
import torch.distributed as dist


def example_to_device(example, device, non_blocking=False):
    example_torch = {}
    for k, v in example.items():
        if k in ['token']:
            example_torch[k] = v
        elif isinstance(v, list):
            example_torch[k] = [
                res.to(device, non_blocking=non_blocking) for res in v]
        else:
            example_torch[k] = v.to(device, non_blocking=non_blocking)

    return example_torch


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_every_niters = log_every_niters

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.epoch = 0
        self.global_step = 0
        self.inner_iter = 0
        self.max_epochs = max_epochs

        self.clip_grad_val = clip_grad_val
        self.eval_every_nepochs = eval_every_nepochs
        self.eval_epochs = eval_epochs

        self.logger = logger

    @property
    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError(
                "lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def device(self):
        """
        :return: an instance of `torch.device`
        """
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return torch.device('cpu')

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict)

    def save_checkpoint(self, filename_tmpl="epoch_{}.pth", save_optimizer=True):
        meta = dict(epoch=self.epoch, iter=self.global_step)
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer,
                        scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        checkpoint = self.load_checkpoint(
            checkpoint, map_location=map_location, strict=True)

        self.epoch = checkpoint['meta']['epoch']
        self.global_step = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and resume_optimizer:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        self.logger.info("resumed epoch %d, iter %d",
                         self.epoch, self.global_step)

    def optimize_step(self, loss):
        """
        Performs a single optimization step and updates the parameters which have been passed to ``self.optimizer``.
        """
        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_val)

        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train_iter(self, data_batch):
        data_batch = example_to_device(
            data_batch, self.device, non_blocking=False)
        loss, logs = self.model(data_batch)
        self.optimize_step(loss)

        if (self.inner_iter + 1) % self.log_every_niters == 0:
            log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
                self.epoch + 1,
                self.max_epochs,
                self.inner_iter + 1,
                len(self.train_dataloader),
                self.current_lr[0])
            self.logger.info(log_str)
            self.logger.info(self._convert_to_str(logs))

        # update global step
        self.global_step += 1

    def train_epoch(self):
        self.model.train()
        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(self.epoch)

        for i, data_batch in enumerate(self.train_dataloader):
            self.inner_iter = i  # iteration in current epoch
            self.train_iter(data_batch)

        self.epoch += 1
        if self.rank == 0:
            self.save_checkpoint()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()

        if self.rank == 0:
            prog_bar = ProgressBar(len(self.val_dataloader))

        results = {}

        for i, data_batch in enumerate(self.val_dataloader):
            self._inner_iter = i
            data_batch = example_to_device(
                data_batch, torch.cuda.current_device(), non_blocking=False)
            res = self.model(data_batch)
            results.update(res)
            if self.rank == 0:
                prog_bar.update()

        # gather results across gpu
        if self.world_size > 1:
            dist.barrier()
            all_predictions = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_predictions, results)

        if self.rank != 0:
            return

        if self.world_size > 1:
            predictions = {}
            for p in all_predictions:
                predictions.update(p)
        else:
            predictions = results

        # after_val_epoch()
        output_dir = Path("results") / f"epoch_{self.epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dict = self.val_dataloader.dataset.evaluation(
            predictions, output_dir)

        self.logger.info("\n")
        for k, v in result_dict.items():
            self.logger.info(f"Evaluation {k}: {v}")

    def fit(self):
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            if (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0) or \
               (self.eval_epochs is not None and self.epoch in self.eval_epochs):
                self.val_epoch()

    def _convert_to_str(self, log_dict):
        def _convert_to_precision4(val):
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().tolist()
            if isinstance(val, float):
                val = "{:.4f}".format(val)
            elif isinstance(val, list):
                val = [_convert_to_precision4(v) for v in val]

            return val

        def _convert_dict_to_str(log_vars):
            log_items = []
            for name, val in log_vars.items():
                log_items.append("{}: {}".format(
                    name, _convert_to_precision4(val)))

            log_str = ", ".join(log_items)

            return log_str

        if isinstance(log_dict, list):
            logs = [_convert_dict_to_str(log_vars) for log_vars in log_dict]
            log_str = '\n'.join(logs)
        else:
            log_str = _convert_dict_to_str(log_dict)

        log_str += "\n"

        return log_str
