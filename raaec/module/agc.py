import logging

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

class DummyGAIN(nn.Module):
    def __init__(self):
        super().__init__()
        logging.warning("using DummyGAIN")

    def forward(self, x):
        return x

def init_agc(agc_conf):
    agc_select = agc_conf.get('select', 'FIXGAIN')
    if agc_select == 'DummyGAIN':
        return DummyGAIN()
    agc_class = eval(agc_select)
    return agc_class(**agc_conf['conf'])
