import os
import logging

from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils.data as tdata
import pytorch_lightning as pl

import raaec.data.dataset as dataset
from raaec.data.dataset import AECCDATASET
from raaec.data.dataset import WAVDIRDATASET
from raaec.data.dataset import TIMITDATASET
from raaec.data.dataset import TIMIT_FLTER_AECDATASET

from raaec.utils.set_config import hydra_runner

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class Pad_tensor(object):
    def __init__(self, pad, dim):
        self.pad = pad
        self.dim = dim

    def __call__(self, vec):
        return pad_tensor(vec, self.pad, self.dim)


class SinglePadCollate(object):
    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        # data extract
        each_data_len = [x.shape[self.dim] for x in batch]
        max_len = max(each_data_len)
        padded_each_data = [pad_tensor(x, max_len, self.dim) for x in batch]
        data = torch.stack(padded_each_data, dim=0)
        data_len = torch.tensor(each_data_len)
        return data, data_len

    def __call__(self, batch):
        return self.pad_collate(batch)


class MulPadCollate(object):
    def __init__(self, pad_choices, dim=0):
        super().__init__()
        self.dim = dim
        self.pad_choices = pad_choices

    def pad_collate(self, batch):
        # data extract
        data = list()
        data_len = list()
        for i, pad_choice in enumerate(self.pad_choices):
            if pad_choice:
                each_data = [x[i] for x in batch]
                each_data_len = [x.shape[self.dim] for x in each_data]
                max_len = max(each_data_len)
                padded_each_data = [pad_tensor(x, max_len, self.dim) for x in each_data]
                data.append(torch.stack(padded_each_data, dim=0))
                data_len.append(torch.tensor(each_data_len))
        return data, data_len

    def __call__(self, batch):
        return self.pad_collate(batch)

class TrainDataModule(pl.LightningDataModule):
    # only support aecc data right now
    # TODO: support timit
    def __init__(
            self, data_setting, data_loader_setting):
        super().__init__()
        self.data_setting = data_setting
        self.data_loader_setting = data_loader_setting

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage=None):
        datasets = []
        if stage == "train":
            if "aecc" in self.data_setting['trainset']:
                aeccdataset = AECCDATASET(**self.data_setting['aecc'])
                print(f"the length of aecc data is {len(aeccdataset)}")
                datasets.append(aeccdataset)

            if "timitfilter" in self.data_setting['trainset']:
                raise NotImplementedError('have not implemented yet')
                filterdataset = WAVDIRDATASET(**self.data_setting['timitfilter']['filter'])
                print(f"the length of filter data is {len(filterdataset)}")
                timitdataset = TIMITDATASET(**self.data_setting['timitfilter']['timit'])
                print(f"the length of timit data is {len(timitdataset)}")
                timit_filter_dataset = TIMIT_FLTER_AECDATASET(timitdataset, filterdataset)
                print(f"the length of timit_filter data is {len(timit_filter_dataset)}")
            self.dataset = tdata.ConcatDataset(datasets)
            self.train_collect_fn = MulPadCollate(
                    pad_choices=[True] * 2, dim=0)

    def train_dataloader(self):
        return tdata.DataLoader(
            self.dataset, **self.data_loader_setting,
             collate_fn=self.train_collect_fn)

def init_datamodule(data_conf):
    return TrainDataModule(data_conf['dataset'], data_conf['data_loader'])

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    dm = init_datamodule(cfg['data'])
    print(f"datamodule: {dm}")
    # dm.setup('train')
    # train_dataloader = dm.train_dataloader()
    # print(f"len of the dataloader {len(train_dataloader)}")

if __name__ == "__main__":
    unit_test()