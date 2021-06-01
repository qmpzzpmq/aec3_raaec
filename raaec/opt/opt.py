import logging

import torch
import pytorch_lightning as pl

from raaec.model.raaec import init_model

class RAAEC(pl.LightningModule):
    def __init__(self, raaec_model, opt,):
        super().__init__()
        self.raaec_model = raaec_model
        self.opt = opt

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        return self.opt

def init_opt(opt_conf):
    opt_policy = opt_conf['builder_policy']
    logging.warning(f"using {builder_policy} builder")

if __name__ == "__main__":
    raaec_model = init_model()
    RAAEC(raaec_model, opt)