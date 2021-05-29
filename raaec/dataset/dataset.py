import os
import logging

import librosa
import torch
import torch.nn as nn
import torchaudio
import numpy as np

import torch.utils.data as tdata
import timit_utils as tu

from raaec.DSP.common_DSP import wav_norm

class wavdirdataset(tdata.Dataset):
    def __init__(
            self, datain, fs=16000, read_dur=None, wav_norm=False):
        super().__init__()
        self.audiolist = list()
        self.fs = fs
        self.read_dur = read_dur if read_dur is not None else float("Inf")
        self.wav_norm = wav_norm
        if os.path.isdir(datain):
            for filename in os.listdir(datain):
                if "." not in filename:
                    continue
                _, ext = filename.rsplit(".", 1)
                if ext != "mp3" and ext != "wav" and ext != "flac":
                    continue
                self.audiolist.append(os.path.join(datain, filename))
        else:
            for filename in open(datain, "r"):
                filename = filename.strip()
                _, ext = filename.rsplit(".", 1)
                if ext != "mp3" and ext != "wav" and ext != "flac":
                    continue
                self.audiolist.append(os.path.join(filename))

    def __getitem__(self, index):
        si, _ = torchaudio.info(self.audiolist[index])
        num_frames = min(si.rate*self.read_dur, si.length)
        data, _ = torchaudio.load(
            self.audiolist[index], normalization=self.wav_norm,
            num_frames=num_frames)
        return data[0, :]

    def __len__(self):
        return len(self.audiolist)

class timitdataset(tdata.Dataset):
    def __init__(
            self, subset, dirpath, fs, wav_norm=False):
        assert os.path.isdir(dirpath)
        super().__init__()
        self.audiolist = list()
        self.wav_norm = wav_norm
        corpus = tu.Corpus(dirpath)
        if subset == "train":
            subset = corpus.train
        elif subset == "test":
            subset = corpus.test
        else:
            raise ValueError(f"no {subset} in TIMIT dataset")

        for p_name, people in subset.people.items():
            for s_name, sentence in people.sentences.items():
                data = sentence.raw_audio.astype(np.float32)
                data = librosa.resample(librosa.to_mono(data), 16000, fs)
                data = wav_norm(data) if self.wav_norm else data
                self.audiolist.append(
                    {"p_name": p_name, "s_name": s_name, \
                        "data": torch.as_tensor(data)})
        self.num_wav = len(self.audiolist)

    def __getitem__(self, index):
        return self.audiolist[index]['data']

    def __len__(self):
        return self.num_wav

class aecdataset(tdata.Dataset):
    def __init__(
            self, spkdata, fltdata):
        super().__init__()
        if len(spkdata) % 2 == 0:
            self.datalen = int(len(spkdata) / 2)
            self.refdata, self.neardata = tdata.random_split(
                spkdata, [self.datalen, self.datalen])
        else:
            self.datalen = len(spkdata) // 2
            self.refdata, self.neardata, _ = tdata.random_split(
                spkdata, [self.datalen, self.datalen, 1])
        self.fltdata = fltdata

        self.datalen = \
            len(self.neardata) * len(self.refdata) \
            * len(self.fltdata)

    def __getitem__(self, index):
        logging.debug(f"reading test data from {index}")
        fltidx = index // (len(self.refdata) * len(self.neardata))
        refidx = (index // len(self.neardata)) % len(self.refdata)
        nearidx = index % len(self.neardata)

        neardata = self.neardata[nearidx]
        refdata = self.refdata[refidx]
        fltdata = self.fltdata[fltidx]

        return refdata, fltdata, neardata

    def __len__(self):
        return self.datalen

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