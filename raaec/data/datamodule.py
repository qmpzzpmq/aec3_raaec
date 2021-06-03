import logging

import pytorch_lightning as pl
import torch.utils.data as tdata

import raaec.data.dataset as dataset
from raaec.data.dataset import MulPadCollate
from raaec.data.dataset import SinglePadCollate


class TrainDataModule(pl.LightningDataModule):
    def __init__(
            self, ref_dir, flt_dir, fs, batch_size, train_shuffle, val_shuffle,
            num_workers, near_data_for_train=False, is_ddp=False):
        super().__init__()
        self.ref_dir = ref_dir
        self.flt_dir = flt_dir
        self.fs = fs
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.num_workers = num_workers
        self.is_ddp = is_ddp
        if near_data_for_train:
            self.train_collect_fn = MulPadCollate(
                pad_choices=[True] * 3, dim=0)
        else:
            self.train_collect_fn = MulPadCollate(
                pad_choices=[True] * 2, dim=0)
        self.val_collect_fn = MulPadCollate(pad_choices=[True] * 3, dim=0)
        self.near_data_for_train = near_data_for_train

    def prepare_data(self):
        pass

    def setup(self, stage):
        if self.ref_dir.split("/")[-1] == "TIMIT":
            self.train_refdata = dataset.timitdataset(
                subset="train", dirpath=self.ref_dir, fs=self.fs)
            self.test_refdata = dataset.timitdataset(
                subset="test", dirpath=self.ref_dir, fs=self.fs)
        else:
            self.train_refdata = dataset.wavdirdataset(
                self.ref_dir, fs=self.fs)

        fltdata = dataset.wavdirdataset(
            self.flt_dir, fs=self.fs, read_dur=0.8)
        fltdat_len = len(fltdata)
        half_fltdata_len = fltdat_len // 2
        self.train_fltdata, self.test_fltdata = tdata.random_split(
            fltdata, [half_fltdata_len, fltdat_len - half_fltdata_len])

    def train_dataloader(self):
        if self.near_data_for_train:
            train_simudata = dataset.testdataset(
                self.train_refdata, self.train_fltdata)
        else:
            train_simudata = dataset.simudataset(
                self.train_refdata, self.train_fltdata)
        logging.debug(f"the lenght of train dataset is {len(train_simudata)}")
        if self.is_ddp and self.train_shuffle:
            sampler = tdata.distributed.DistributedSampler(
                train_simudata, shuffle=True)
        else:
            sampler = None
        train_dataloader = tdata.DataLoader(
            train_simudata, batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_collect_fn,
            sampler=sampler)
        return train_dataloader

    def val_dataloader(self):
        test_simudata = dataset.testdataset(
            self.test_refdata, self.test_fltdata)
        logging.debug(f"the lenght of val dataset is {len(test_simudata)}")
        if self.is_ddp and self.val_shuffle:
            sampler = tdata.distributed.DistributedSampler(
                test_simudata, shuffle=True)
        else:
            sampler = None
        test_dataloader = tdata.DataLoader(
            test_simudata, batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_collect_fn,
            sampler=sampler)
        return test_dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
            self, datain, fs, batch_size, shuffle, num_workers):
        super().__init__()
        self.fs = fs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collect_fn = SinglePadCollate(dim=0)
        if datain.split("/")[-1] == "TIMIT":
            self.wholedataset = dataset.timitdataset(
                subset="train", dirpath=datain, fs=self.fs)
        else:
            self.wholedataset = dataset.wavdirdataset(datain, fs=self.fs)

    def prepare_data(self):
        pass

    def setup(self, stage):
        valsetlen = int(len(self.wholedataset) * 0.1)
        valsetlen = min(valsetlen, 2000)
        self.traindata, self.valdata = tdata.random_split(
            self.wholedataset,
            [valsetlen, len(self.wholedataset)-valsetlen])

    def train_dataloader(self):
        train_dataloader = tdata.DataLoader(
            self.traindata, batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collect_fn)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = tdata.DataLoader(
            self.valdata, batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collect_fn)
        return val_dataloader