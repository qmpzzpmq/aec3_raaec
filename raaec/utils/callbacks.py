import os
import logging

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks

from raaec.utils.set_config import hydra_runner

def init_callbacks(callbacks_conf):
    callbacks = []
    if callbacks_conf.get("progressbar", False):
        callbacks.append(
            pl_callbacks.progress.ProgressBar(
                **callbacks_conf['progressbar_conf']
            )
        )

    if callbacks_conf.get("modelcheckpoint", False):
        callbacks.append(
            pl_callbacks.model_checkpoint.ModelCheckpoint(
                **callbacks_conf['modelcheckpoint_conf']
            )
        )
    
    if callbacks_conf.get("earlystopping", False):
        callbacks.append(
            pl_callbacks.early_stopping.EarlyStopping(**callbacks_conf['earlystopping_conf'])
        )

    return callbacks

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    callbacks = init_callbacks(cfg['callbacks'])
    print(f"callbacks: {callbacks}")

if __name__ == "__main__":
    unit_test()
