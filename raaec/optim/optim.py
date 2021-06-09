import os
import logging

from omegaconf import DictConfig, OmegaConf
import torch

from raaec.utils.set_config import hydra_runner
from raaec.module.aec_module import init_module

def init_optim(model, optim_conf):
    optim_select = optim_conf.get('optim_select', "adam")
    logging.warning(f"using {optim_select} builder")
    if optim_select == "sgd":
        return torch.optim.SGD(model.parameters(), **optim_conf['optim_conf'])
    elif optim_select == "adam":
        return torch.optim.Adam(model.parameters(), **optim_conf['optim_conf'])
    else:
        raise NotImplementedError(f"the optim policy {optim_select}")

def init_scheduler(optim, scheduler_conf):
    scheduler_select = scheduler_conf.get('scheduler_select', "lambdalr")
    if scheduler_select == "lambdalr":
        lr_lambda = lambda epoch: 0.95 ** epoch
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lr_lambda,
            **scheduler_conf['scheduler_conf']
        )
    else:
        raise NotImplementedError(f"the scheduler policy {scheduler_select}")  

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    raaec = init_module(cfg['module'], cfg['optim'])
    print(f"raaec: {raaec}")

if __name__ == "__main__":
    unit_test()
