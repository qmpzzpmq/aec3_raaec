import os
import logging

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.utils.data as tdata

import raaec.data.dataset as dataset
from raaec.data.dataset import MulPadCollate
from raaec.data.dataset import SinglePadCollate
from raaec.data.dataset import AECCDATASET
from raaec.data.dataset import WAVDIRDATASET
from raaec.data.dataset import TIMITDATASET
from raaec.data.dataset import TIMIT_FLTER_AECDATASET

from raaec.utils.set_config import hydra_runner

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


@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    dm = TrainDataModule(cfg['data']['dataset'], cfg['data']['data_loader'])
    dm.setup()
    train_dataloader = dm.train_dataloader()
    print(f"len of the dataloader {len(train_dataloader)}")

if __name__ == "__main__":
    unit_test()