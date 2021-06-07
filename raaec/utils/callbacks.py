import os

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
        print(f"model check point config {callbacks_conf['modelcheckpoint_conf']}")
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
    print(f"config: {cfg}")
    init_callbacks(cfg['callbacks'])

if __name__ == "__main__":
    unit_test()
