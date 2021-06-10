import torch
import torch.nn as nn
import torchaudio as ta

from raaec.module.agc import init_agc

class FRONTEND(nn.Module):
    def __init__(self, fft_conf, agc_conf):
        super().__init__()
        self.fft = ta.transforms.Spectrogram(**fft_conf['fft_conf'], power=2)
        self.agc = init_agc(agc_conf)

    def forward(self, signals):
        signals_power = []
        for signal in signals:
            signals_power.append(
                self.fft(self.agc(signal)).transpose(-1, -2)
            )
        return signals_power

def init_frontend(frontend_conf):
    return FRONTEND(frontend_conf['fft'], frontend_conf['agc'])