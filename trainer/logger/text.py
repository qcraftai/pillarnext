from trainer.trainer.callbacks import TrainerCallback
import torch

class TextLogger(TrainerCallback):
    """
    TextLogger
    """
    def __init__(self, log_every_niters):
        super(TextLogger, self).__init__()
        self.log_every_niters = log_every_niters

    def after_train_iter(self, trainer):
        if (trainer.inner_iter + 1) % self.log_every_niters == 0:
            self.meta_log(trainer)

            trainer.logger.info(self._convert_to_str(trainer.train_logs))
    
    def meta_log(self, trainer):
        log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
                trainer.epoch + 1,
                trainer.max_epochs,
                trainer.inner_iter + 1,
                len(trainer.train_dataloader),
                trainer.current_lr[0])
        trainer.logger.info(log_str)


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
                log_items.append("{}: {}".format(name, _convert_to_precision4(val)))

            log_str = ", ".join(log_items)

            return log_str
        
        
        if isinstance(log_dict, list):
            logs = [_convert_dict_to_str(log_vars) for log_vars in log_dict]
            log_str = '\n'.join(logs)
        else:
            log_str = _convert_dict_to_str(log_dict)
            
        log_str += "\n"

        return log_str

    def after_val_epoch(self, trainer):
        pass