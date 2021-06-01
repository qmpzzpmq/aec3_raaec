import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio as ta

from raaec.aec3.webrtc_aec3 import AEC3
from raaec.DSP.torch_DSP import lengths_sub

class Frontend(nn.Module):
    def __init__(self, n_fft=512, hop_length=400):
        super().__init__()
        self.fft = ta.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1)

    def forward(self, est, ref):
        lengths_sub(est, ref)
        est_energy = self.fft(est)
        ref_energy = self.fft(ref)
        return torch.cat([est_energy, ref_energy], dim=-1).log10()

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = [(x - 1) // 2 for x in kernel_size] \
            if type(kernel_size) == tuple or type(kernel_size) == list \
            else (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class AEC_InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, norm_layer=None):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = [
            # pw
            ConvBNReLU(inp, inp, kernel_size=1, norm_layer=norm_layer),
            # dw
            ConvBNReLU(inp, inp, kernel_size=kernel_size, stride=stride, groups=inp, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MASKS_DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            AEC_InvertedResidual(128, 64, 3),
            ConvBNReLU(128, 64, (3,4)),
            nn.Linear(64, 64),
        )
    def forward(self, x):
        h = self.main(x)
        return F.sigmoid(h), h 

class DTD_DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AEC_InvertedResidual(128, 24, (3,4))
        self.dec = nn.Linear(88, 3)
    def forward(self, x, condition):
        h = self.enc(x)
        out = self.dec(torch.cat((h, condition), dim=-1))
        return F.softmax(x)

class RAAEC_MODEL(nn.Module):
    def __init__(
            self, frontend, af,
        ) -> None:
        super().__init__()
        self.frontend = frontend
        self.af = af
        enc_channels = [32, 64, 64, 128, 128]
        enc_strides = [2, 1, 2, 1]
        assert len(enc_channels) - 1 == len(enc_strides)
        enc_layers = []
        enc_layers.append(nn.Conv2d(1, 32, 3))
        for i in range(len(enc_channels) - 1):
            enc_layers.append(
                AEC_InvertedResidual(enc_channels[i], enc_channels[i+1], 3, enc_strides[i])
            )
        self.enc = nn.Sequential(*enc_layers)
        self.masks_dec = MASKS_DEC()
        self.DTD_dec = DTD_DEC()

    def forward(self, ref, rec):
        est, _, = self.aec3.linear_run(ref, rec)
        x = self.frontend(est, ref)
        h = self.enc(x)
        masks, condition = self.masks_dec(h)
        if not self.training:
            return masks
        DTD = self.DTD_dec(h, condition)
        return masks, DTD

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def init_model(n_fft=512, hop_length=400, fs=16000):
    return RAAEC_MODEL(
        frontend=Frontend(n_fft=n_fft, hop_length=hop_length),
        af=AEC3(fs=fs, pure_linear=True),
    )

if __name__ == "__main__":
    raaec = init_model(n_fft=512, hop_length=400, fs=16000)
    raaec.train()
    ref = ta.load('ref.wav')
    rec = ta.load('ref.wav')
    masks, DTD = raaec(ref, rec)
