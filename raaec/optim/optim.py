import os
import logging

from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from contiguous_params import ContiguousParams

def init_optim(model, optim_conf):
    optim_select = optim_conf.get('select', "optim.Adam")
    logging.warning(f"using {optim_select} builder")
    optim_class = eval(optim_select)
    parameters = ContiguousParams(model.parameters()) \
        if optim_conf.get('contiguous_params', False) \
        else model.parameters()
    return optim_class(parameters, **optim_conf['conf'])

def init_scheduler(optim, scheduler_conf):
    scheduler_select = scheduler_conf.get('scheduler_select', "lambdalr")
    if scheduler_select == "lambdalr":
        scheduler_conf = dict(scheduler_conf['conf'])
        lr_lambda = scheduler_conf.pop('lr_lambda')
        lr_lambda = eval(lr_lambda)
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lr_lambda,
            **scheduler_conf,
        )
    else:
        raise NotImplementedError(f"the scheduler policy {scheduler_select}")
