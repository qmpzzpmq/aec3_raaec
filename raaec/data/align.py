import os
import logging

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.data as tdata

from raaec.utils.set_config import hydra_runner
# import raaec.data.dataset as local_dataset
from raaec.module.agc import init_agc

def cross_correlation(x, y):
    return F.conv1d(x.view(1, 1, -1), y.view(1, 1, -1), padding=y.size(-1)-1).view(-1)

def signal_select(signals, select):
    if select == len(signals):
        return signals[select], signals[:select]
    elif select == 0:
        return signals[0], signals[1:]
    else:
        signals[select], signals[:select] + signals[select:]

def correlation_index(signals, abs=False):
    if abs:
        signals = [x.abs() for x in signals]
    lens = []
    for signal in signals:
        lens.append(signal.size(dim=-1))
    min_len, min_idx = torch.tensor(lens).min(dim=0)
    time_idxs = []
    for i, signal in enumerate(signals):
        if i == min_idx:
            time_idxs.append(0)
        else:
            time_idx = cross_correlation(
                signals[min_idx], signal).squeeze().argmax() - (signals[min_idx].size(-1)+1 //2)
            time_idxs.append(time_idx.item())
    return min_len, time_idxs

def time_correlation_align(signals, time_idxs, min_len):
    time_align_signals = []
    for signal, time_idx in zip(signals, time_idxs):
        if time_idx == 0:
            time_align_signal = signal[0:min_len]
        else:
            time_idx = abs(time_idx)
            time_align_signal = signal[time_idx:min_len]
        time_align_signals.append(time_align_signal)
    return time_align_signals

def correlation_align(signals, abs=False):
    min_len, time_idxs = correlation_index(signals, abs=abs)
    time_align_signal = time_correlation_align(signals, time_idxs, min_len)
    return time_align_signal


# @hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
# def unit_test(cfg: DictConfig):
#     logging.basicConfig(
#         level=eval(f"logging.{cfg['logging'].get('level', 'INFO')}"),
#         format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
#     logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
# 
#     datasets = []
#     for dataset in cfg['data']['dataset']['train']:
#         dataset_class = getattr(local_dataset, dataset['select'])
#         dataset['conf']['length_align'] = False
#         datasets.append(
#             dataset_class(**dataset['conf'])
#         )
#     dataset = tdata.ConcatDataset(datasets)
#     agc = init_agc(cfg['module']['conf']['frontend']['agc'])
# 
#     signals = dataset[0]
#     signals = [agc(x).unsqueeze(0).cuda() for x in signals]
#     min_len, time_idxs, magnitute_idxs = correlation_index(signals)
#     print(f"time index: {time_idxs}")
#     print(f"magnitute index: {magnitute_idxs}")
# 
# if __name__ == "__main__":
#     unit_test()
