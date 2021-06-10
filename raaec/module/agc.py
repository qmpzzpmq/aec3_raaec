import torch
import torch.nn as nn

# TODO: a real AGC
class FIXGAIN(nn.Module):
    def __init__(self, gain_index):
        super().__init__()
        self.gain_index = gain_index

    def forward(self, x):
        x_gain = x / self.gain_index - 1
        return x_gain.to(dtype=torch.float)

def init_agc(agc_conf):
    agc_class = eval(f"{agc_conf['select']}")
    return agc_class(**agc_conf['agc_conf'])
