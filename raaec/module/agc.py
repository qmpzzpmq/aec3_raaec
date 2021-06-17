import torch
import torch.nn as nn

# TODO: a real AGC
class FIXGAIN(nn.Module):
    def __init__(self, gain_index=32768):
        super().__init__()
        self.gain_index = gain_index

    def forward(self, x):
        x_gain = x / self.gain_index
        return x_gain.float()

def init_agc(agc_conf):
    agc_class = eval(f"{agc_conf.get('select', 'FIXGAIN')}")
    return agc_class(**agc_conf['agc_conf'])
