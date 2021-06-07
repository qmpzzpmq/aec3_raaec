import logging
import os
from typing import Any, Callable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from raaec.module.mobilenet import RAAEC_MODEL
from raaec.utils.set_config import hydra_runner

def init_module(module_conf):
    module_select = module_conf.get('module_select', "mobilenet")
    if module_select == 'mobilenet':
        return RAAEC_MODEL(**module_conf['module_conf'])
    else:
        raise NotImplementedError(f"{module_select} haven't been implemented")

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    module = init_module(cfg['module'])
    print(f"module {module}")

if __name__ == "__main__":
    unit_test()