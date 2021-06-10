import logging
import os
from typing import Any, Callable, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from raaec.module.mobilenet import AEC_MOBILENET
from raaec.module.loss import init_loss
from raaec.optim.optim import init_optim
from raaec.optim.optim import init_scheduler
from raaec.data.datamodule import singlepadcollate
from raaec.utils.set_config import hydra_runner

def init_module(module_conf):
    module_select = module_conf.get('select', "mobilenet")
    if module_select == 'mobilenet':
        return AEC_MOBILENET(**module_conf['module_conf'])
    else:
        raise NotImplementedError(f"{module_select} haven't been implemented")

class RAAEC(pl.LightningModule):
    def __init__(self, module_conf, optim_conf=None, loss_conf=None):
        super().__init__()
        self.raaec = init_module(module_conf)
        if optim_conf is not None:
            self.optim = init_optim(self.raaec, optim_conf)
            self.scheduler = init_scheduler(self.optim, optim_conf)
        if loss_conf is not None:
            self.loss = init_loss(loss_conf)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss = self.loss_compute(batch)
        for k, v in loss.items():
            self.log(
                f"train_{k}", v, on_step=True, on_epoch=False, 
                prog_bar=True, logger=True,
            )
            loss['f"train_{k}"'] = loss.pop(k)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_compute(batch)
        for k, v in loss.items():
            self.log(
                f"val_{k}", v, on_step=True, on_epoch=True, 
                prog_bar=True, logger=True,
            )
            loss['f"train_{k}"'] = loss.pop(k)
        return loss

    def loss_compute(self, batch):
        datas, datas_len = batch
        refs, recs, nears = datas

        predict_masks = []
        predict_DTDs = []
        for ref, rec, near in zip(refs, recs, nears):
            predict_mask, predict_DTD, est_power, ref_power = self.raaec(
                ref.squeeze(), rec.squeeze())
            predict_masks.append(predict_mask)
            predict_DTDs.append(predict_DTD)
        pad_predict_masks, masks_len = singlepadcollate(predict_masks)
        pad_predict_DTDs, DTDs_len = singlepadcollate(predict_DTDs)

        recs_power, nears_power = self.raaec.frontend([recs, nears])
        return self.loss(
            pad_predict_DTDs, pad_predict_masks, recs_power, nears_power, masks_len.sum()
        )

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