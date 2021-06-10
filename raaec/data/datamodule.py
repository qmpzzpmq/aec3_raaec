import os
import logging
import importlib

from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils.data as tdata
import pytorch_lightning as pl

from raaec.utils.set_config import hydra_runner
import raaec.data.dataset as local_dataset

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
    return torch.cat(
        [vec, torch.zeros(
            *pad_size, dtype=vec.dtype, device=vec.device)]
        , dim=dim
    )

def singlepadcollate(batch, dim=-1):
    each_data_len = [x.shape[dim] for x in batch]
    max_len = max(each_data_len)
    padded_each_data = [pad_tensor(x, max_len, dim) for x in batch]
    data = torch.stack(padded_each_data, dim=0)
    data_len = torch.tensor(each_data_len)
    return data, data_len

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
        if stage in (None, 'fit'):
            datasets = []
            for dataset in self.data_setting['train']:
                dataset_class = getattr(local_dataset, dataset['select'])
                datasets.append(
                    dataset_class(**dataset['conf'])
                )
            self.train_dataset = tdata.ConcatDataset(datasets)
            self.train_collect_fn = MulPadCollate(
                    pad_choices=[True] * 3, dim=0)
            
            datasets = []
            for dataset in self.data_setting['val']:
                dataset_class = getattr(local_dataset, dataset['select'])
                datasets.append(
                    dataset_class(**dataset['conf'])
                )
            self.val_dataset = tdata.ConcatDataset(datasets)
            self.val_collect_fn = MulPadCollate(
                    pad_choices=[True] * 3, dim=0)

        if stage in (None, 'test'):
            datasets = []
            for dataset in self.data_setting['test']:
                dataset_class = getattr(local_dataset, dataset['select'])
                datasets.append(
                    dataset_class(**dataset['conf'])
                )
            self.test_dataset = tdata.ConcatDataset(datasets)
            self.test_collect_fn = MulPadCollate(
                    pad_choices=[True] * 3, dim=0)

    def train_dataloader(self):
        return tdata.DataLoader(
            self.train_dataset, **self.data_loader_setting,
            collate_fn=self.train_collect_fn)

    def val_dataloader(self):
        return tdata.DataLoader(
            self.val_dataset, **self.data_loader_setting,
            collate_fn=self.val_collect_fn)

    def test_dataloader(self):
        return tdata.DataLoader(
            self.val_dataset, **self.data_loader_setting,
            collate_fn=self.val_collect_fn)
    

def init_datamodule(data_conf):
    return TrainDataModule(data_conf['dataset'], data_conf['data_loader'])

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    dm = init_datamodule(cfg['data'])
    print(f"datamodule: {dm}")
    # dm.setup('train')
    # train_dataloader = dm.train_dataloader()
    # print(f"len of the dataloader {len(train_dataloader)}")

if __name__ == "__main__":
    unit_test()