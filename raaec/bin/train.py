import os
import logging

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.utils.data as tdata

from raaec.utils.callbacks import init_callbacks
from raaec.utils.logger import init_loggers
from raaec.module.aec_module import RAAEC
from raaec.data.datamodule import init_datamodule
from raaec.utils.set_config import hydra_runner


@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="train")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=eval(f"logging.{cfg['logging'].get('level', 'INFO')}"),
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    pl.seed_everything(cfg['base']['seed'], workers=True)

    callbacks = init_callbacks(cfg['callbacks'])
    loggers = init_loggers(cfg['loggers'])
    raaec = RAAEC(cfg['module'], cfg['optim'], cfg['loss'])
    dm = init_datamodule(cfg['data'])

    trainer_conf = cfg['trainer']
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        **trainer_conf,
    )
    trainer.fit(raaec, datamodule=dm)

if __name__ == "__main__":
    main()
