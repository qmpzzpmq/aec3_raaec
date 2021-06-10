import os
from typing import Any, Callable, Optional

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from raaec.DSP.torch_DSP import DTD_compute
from raaec.utils.set_config import hydra_runner

class MASK_DTD_LOSS(nn.Module):
    def __init__(self, criterion_DTD, criterion_mask, DTDweight):
        super().__init__()

        loss_class = eval(criterion_DTD)
        self.loss_DTD = loss_class(reduction="sum")

        loss_class = eval(criterion_mask)
        self.loss_mask = loss_class(reduction="sum")
        self.DTDweight = DTDweight

    def forward(self, pad_predict_DTDs, pad_predict_masks, recs_power, nears_power):
        loss_DTD = self.loss_DTD(pad_predict_DTDs, DTD_compute(recs_power, nears_power))
        # TODO: IAM -> PSM
        loss_mask = self.loss_mask(pad_predict_masks, nears_power / pad_predict_masks)
        return loss_mask + self.DTDweight * loss_DTD

def init_loss(loss_conf):
    loss_class = eval(loss_conf.get('select', 'nn.BCELoss'))
    return loss_class(**loss_conf['loss_conf'])

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    loss = init_loss(cfg['loss'])
    print(f"loss: {loss}")

if __name__ == "__main__":
    unit_test()
