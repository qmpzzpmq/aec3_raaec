import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl

from raaec.model.mobilenet import init_model
from raaec.utils.set_config import hydra_runner

class RAAEC(pl.LightningModule):
    def __init__(self, raaec_model, opt,):
        super().__init__()
        self.raaec_model = raaec_model
        self.opt = opt

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        return self.opt

def init_optim(model, optim_conf):
    optim_policy = optim_conf.get('opt_policy', "adam")
    logging.warning(f"using {optim_policy} builder")
    if optim_policy == "sgd":
        return torch.optim.SGD(model.parameters(), **optim_conf['optim_conf'])
    elif optim_policy == "adam":
        return torch.optim.Adam(model.parameters(), **optim_conf['optim_conf'])
    else:
        raise NotImplementedError(f"the optim policy {optim_policy}")

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    raaec_model = init_model()
    optim = init_optim(raaec_model, cfg.optim)
    raaec = RAAEC(raaec_model, optim)

if __name__ == "__main__":
    unit_test()
