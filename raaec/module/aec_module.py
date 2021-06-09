import logging
import os
from typing import Any, Callable, Optional
import importlib

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from raaec.module.mobilenet import RAAEC_MODEL
from raaec.optim.optim import init_optim
from raaec.optim.optim import init_scheduler
from raaec.data.datamodule import pad_tensor
from raaec.utils.set_config import hydra_runner

def init_module(module_conf):
    module_select = module_conf.get('module_select', "mobilenet")
    if module_select == 'mobilenet':
        return RAAEC_MODEL(**module_conf['module_conf'])
    else:
        raise NotImplementedError(f"{module_select} haven't been implemented")

def init_loss(loss_conf):
    loss_class = importlib.import_module(loss_conf['select'], package='torch')
    return loss_class(reduction="sum")

class RAAEC(pl.LightningModule):
    def __init__(self, module_conf, optim_conf=None, loss_conf=None):
        super().__init__()
        self.raaec_model = init_module(module_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.raaec_model, optim_conf)
            self.scheduler = init_scheduler(self.optim, optim_conf)
        if loss_conf is not None:
            self.loss = init_loss(loss_conf)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        datas, datas_len = batch
        refs, recs, nears = datas
        predicts = []
        predicts_max_len = 0
        for ref, rec, near in zip(refs, recs, nears):
            predict = self.raaec_model(ref, rec)
            predicts.append(predict)
            predicts_max_len = max(predicts_max_len, predict.size(0))
        pad_predicts = pad_tensor(predicts, predicts_max_len, 0)
        loss = self.loss(pad_predicts, nears)
        loss_mean = loss / datas_len.sum() # for further dynamic batch size
        self.log(
            'training_loss', loss_mean,
            on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss_mean

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        return self.optim, self.scheduler

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    module = init_module(cfg['module'])
    print(f"module {module}")

if __name__ == "__main__":
    unit_test()