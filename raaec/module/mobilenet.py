import logging
import os
from typing import Any, Callable, Optional
from math import ceil

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

from raaec.module.frontend import init_frontend
from raaec.module.init import init_init
from raaec.utils.set_config import hydra_runner
from raaec.aec3.webrtc_aec3 import AEC3


class ConvBNReLU(nn.Sequential):
    def __init__(
            self, in_planes, out_planes,
            kernel_size=(3,3), stride=(1, 1), groups=1, norm_layer=None):
        padding = [ceil((x - y) / 2) for x, y in zip(kernel_size, stride)]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding,
                groups=groups, bias=False
            ),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )

class AEC_InvertedResidual(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=(3, 3), stride=(1, 1), norm_layer=None):
        super().__init__()
        self.stride = stride
        assert stride in [(1, 1), (2, 2)]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = [
            # pw
            ConvBNReLU(inp, inp, kernel_size=(1, 1), norm_layer=norm_layer),
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

class MASK_DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            AEC_InvertedResidual(128, 64, (3, 3)),
            ConvBNReLU(64, 64, (3,4)),
        )
        self.linear = nn.Linear(64, 64)
    def forward(self, x):
        h1 = self.main(x)
        h2 = self.linear(h1.transpose(1, 3)).transpose(1, 3)
        # TODO: might be a bug
        return torch.sigmoid(h2.sum(dim=1)), h2
        # return torch.sigmoid(h2), h2

class DTD_DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AEC_InvertedResidual(128, 24, (3, 4))
        self.dec = nn.Linear(88, 3)
    def forward(self, x, condition):
        h = self.enc(x)
        out = self.dec(torch.cat((h, condition), dim=1).transpose(1,3)).transpose(1,3)
        # TODO: might be a bug
        return F.softmax(out.sum(-1), dim=1).transpose(-1,-2)
        # return F.softmax(out, dim=1)

class AEC_MOBILENET(nn.Module):
    def __init__(
            self, frontend, af, init=None,
        ) -> None:
        super().__init__()
        self.frontend = init_frontend(frontend)
        self.af = AEC3(**af)
        enc_channels = [32, 64, 64, 128, 128]
        enc_strides = [(2, 2), (1, 1), (2, 2), (1, 1)]
        assert len(enc_channels) - 1 == len(enc_strides)
        enc_layers = []
        enc_layers.append(nn.Conv2d(1, 32, 3))
        for i in range(len(enc_channels) - 1):
            enc_layers.append(
                AEC_InvertedResidual(enc_channels[i], enc_channels[i+1], (3, 3), enc_strides[i])
            )
        self.enc = nn.Sequential(*enc_layers)
        self.masks_dec = MASK_DEC()
        self.DTD_dec = DTD_DEC()

        if init is not None:
            init_obj = init_init(init)
            for module in self.modules():
                init_obj(module)

    # design for single inference
    def mask(self, ref, rec):
        est, _, = self.af.run_float(
            ref.cpu().numpy(), rec.cpu().numpy()
        )
        est = torch.as_tensor(est, dtype=torch.float, device=ref.device)
        est_power, ref_power = self.frontend([est, ref])
        x = torch.cat([est_power, ref_power], dim=-1).log10()
        h = self.enc(x.unsqueeze(0).unsqueeze(0))
        mask, condition = self.masks_dec(h)
        return mask, condition, h, est_power, ref_power

    # design for single inference
    def forward(self, ref, rec):
        mask, condition, h, est_power, ref_power = self.mask(ref, rec)
        DTD = self.DTD_dec(h, condition)
        return mask.squeeze(0), DTD.squeeze(0), est_power, ref_power

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    raaec = AEC_MOBILENET(**cfg['module']['conf'])
    raaec.train()
    ref, _ = ta.load('ref.wav', normalize=False)
    rec, _ = ta.load('ref.wav', normalize=False)
    mask, DTD = raaec(ref.squeeze(), rec.squeeze())
    print(f"mask: {mask}, DTD: {DTD}")

if __name__ == "__main__":
    unit_test()
