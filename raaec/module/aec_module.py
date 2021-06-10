import logging
import os
from typing import Any, Callable, Optional
import importlib

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from raaec.module.mobilenet import RAAEC_MODEL
from raaec.module.frontend import FRONTEND
from raaec.optim.optim import init_optim
from raaec.optim.optim import init_scheduler
from raaec.data.datamodule import singlepadcollate
from raaec.utils.set_config import hydra_runner

def init_module(module_conf):
    module_select = module_conf.get('module_select', "mobilenet")
    if module_select == 'mobilenet':
        return RAAEC_MODEL(**module_conf['module_conf'])
    else:
        raise NotImplementedError(f"{module_select} haven't been implemented")

def init_loss(loss_conf):
    loss_class = eval(loss_conf['select'])
    return loss_class(reduction="sum")

class RAAEC(pl.LightningModule):
    def __init__(self, module_conf, frontend_conf, optim_conf=None, loss_conf=None):
        super().__init__()
        self.frontend = FRONTEND(**frontend_conf)
        self.raaec_model = init_module(module_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.raaec_model, optim_conf)
            self.scheduler = init_scheduler(self.optim, optim_conf)
        if loss_conf is not None:
            self.loss = init_loss(loss_conf)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.loss_compute(batch)
        self.log(
            'training_loss', loss,
            on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def loss_compute(self, batch):
        datas, datas_len = batch
        refs, recs, nears = datas
        predicts = []
        predicts_max_len = 0
        for ref, rec in zip(refs, recs):
            predict = self.raaec_model(ref.squeeze(), rec.squeeze())
            predicts.append(predict)
            predicts_max_len = max(predicts_max_len, predict.size(0))
        pad_predicts, predicts_len = singlepadcollate(predicts)
        loss = self.loss(pad_predicts, nears)
        loss_mean = loss / predicts_len.sum() # for further dynamic batch size
        return loss_mean

    def validation_step(self, batch, batch_idx):
        loss = self.loss_compute(batch, batch_idx)
        self.log(
            'training_loss', loss,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss
    # def test_step(self, *args, **kwargs):
    #     return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        return [self.optim], [self.scheduler]

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    raaec = RAAEC(cfg['module'], cfg['optim'], cfg['loss'])
    print(f"module {raaec}")

if __name__ == "__main__":
    unit_test()