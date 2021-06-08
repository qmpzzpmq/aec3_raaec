import os
import logging

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from raaec.utils.set_config import hydra_runner

def init_loggers(loggers_conf):
    loggers = []
    if loggers_conf.get('tensorboard', False):
        loggers.append(
            pl_loggers.TensorBoardLogger(**loggers_conf['tensorboard_conf'])
        )

    if loggers_conf.get('wandb', False):
        loggers.append(
            pl_loggers.wandb.WandbLogger(**loggers_conf['wandb_conf'])
        )

    if loggers_conf.get('neptune',False):
        loggers.append(
            pl_loggers.neptune.NeptuneLogger(**loggers_conf['neptune_conf'])
        )

    return loggers

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    loggers = init_loggers(cfg['loggers'])
    print(f"loggers: {loggers}")

if __name__ == "__main__":
    unit_test()