import os
import logging

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.utils.data as tdata

from raaec.utils.callbacks import init_callbacks
from raaec.utils.logger import init_loggers
from raaec.modul.aec_module import init_module
from raaec.utils.set_config import hydra_runner


@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def main(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    pl.seed_everything(cfg['base']['seed'], workers=True)

    callbacks = init_callbacks(cfg['callbacks'])
    loggers = init_loggers(cfg['loggers'])
    raaec = init_module(cfg['module'])

    trainer = pl.Trainer(
        callbacks=callbacks,
        loggers=loggers,
        **cfg['trainer'],
    )

if __name__ == "__main__":
    main()
