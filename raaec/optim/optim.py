import os
import logging

from omegaconf import DictConfig, OmegaConf
import torch

def init_optim(model, optim_conf):
    optim_select = optim_conf.get('select', "adam")
    logging.warning(f"using {optim_select} builder")
    if optim_select == "sgd":
        return torch.optim.SGD(model.parameters(), **optim_conf['conf'])
    elif optim_select == "adam":
        return torch.optim.Adam(model.parameters(), **optim_conf['conf'])
    else:
        raise NotImplementedError(f"the optim policy {optim_select}")

def init_scheduler(optim, scheduler_conf):
    scheduler_select = scheduler_conf.get('scheduler_select', "lambdalr")
    if scheduler_select == "lambdalr":
        lr_lambda = lambda epoch: 0.95 ** epoch
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lr_lambda,
            **scheduler_conf['conf']
        )
    else:
        raise NotImplementedError(f"the scheduler policy {scheduler_select}")
