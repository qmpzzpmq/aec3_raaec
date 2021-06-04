import os
import logging
import glob

from omegaconf import DictConfig, OmegaConf
import librosa
import torch
import torch.nn as nn
import torchaudio as ta
import numpy as np

import torch.utils.data as tdata
import timit_utils as tu

from raaec.DSP.common_DSP import wav_norm
from raaec.utils.set_config import hydra_runner

class WAVDIRDATASET(tdata.Dataset):
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
                _, ext = os.path.splitext(filename)
                if ext != ".mp3" and ext != ".wav" and ext != ".flac":
                    continue
                self.audiolist.append(os.path.join(datain, filename))
        else:
            for filename in open(datain, "r"):
                filename = filename.strip()
                _, ext = os.path.splitext(filename)
                if ext != ".mp3" and ext != ".wav" and ext != ".flac":
                    continue
                self.audiolist.append(os.path.join(filename))

    def __getitem__(self, index):
        si, _ = ta.info(self.audiolist[index])
        num_frames = min(si.rate*self.read_dur, si.length)
        data, _ = ta.load(
            self.audiolist[index], normalization=self.wav_norm,
            num_frames=num_frames)
        return data[0, :]

    def __len__(self):
        return len(self.audiolist)

class TIMITDATASET(tdata.Dataset):
    def __init__(
            self, subset, dirpath, fs, wav_norm=False):
        super().__init__()
        assert os.path.isdir(dirpath)
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

class AECDATASET(tdata.Dataset):
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

class AECCDATASET(tdata.Dataset):
    def __init__(
            self, path,
            select=[
                'doubletalk',
                'doubletalk_with_movement',
                'farend_singletalk',
                'nearend_singletalk',
                'farend_singletalk_with_movement',
                'sweep'],
            dump_path=None,
    ):
        super().__init__()
        self.audio_pairs = aecc_audio_pair(path, os.listdir(path), select) \
            if os.path.isdir(path) else self.load(path)
        if dump_path is not None:
            self.dump(dump_path)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        ref, _ = ta.load(audio_pair[0], normalize=False)
        rec, _ = ta.load(audio_pair[1], normalize=False)
        return ref, rec

    def __len__(self):
        return len(self.audio_pairs)

    def dump(self, dump_path):
        with open(dump_path, "w") as fw:
            for audio_pair in self.audio_pairs:
                fw.write(f"{audio_pair[0]} {audio_pair[1]}\nvim ")

    def load(self, load_path):
        logging.warn(f"using {load_path} file directly for aec challenge data")
        audio_pairs = []
        for i, line in enumerate(open(load_path, 'r')):
            audio_pair = line.strip().split(' ')
            if len(audio_pair) != 2:
                logging.warn(f"the length data in {i}th line of {load_path} is not 2")
                continue
            audio_pairs.append(audio_pair)
        return audio_pairs

def aecc_audio_pair(dir, audio_list, select):
    audio_items = [x.split("_")[0:-1] for x in audio_list \
        if os.path.splitext(x)[1] in [".mp3", ",flac", ".wav"]]
    for i, x in enumerate(audio_items):
        assert len(x) > 1, f"x: {x}, i: {i}, item: {audio_items[i]}, list: {audio_list[i]}"
    audio_items = [[x[0], "_".join(x[1:])] for x in audio_items]
    audio_pairs = {}
    for audio_item in audio_items:
        if audio_item[1] not in select:
            continue
        if audio_item[0] in audio_pairs:
            audio_pairs[audio_item[0]].append(audio_item[1])
        else:
            audio_pairs[audio_item[0]] = [audio_item[1]]
    out_pairs = []
    for k, vs in audio_pairs.items():
        prefix1 = os.path.join(dir, f"{k}")
        for v in vs:
            prefix2 = f"{prefix1}_{v}"
            out_pair = [f"{prefix2}_lpb.wav", f"{prefix2}_mic.wav"]
            if not (os.path.isfile(out_pair[0]) and os.path.isfile(out_pair[1])):
                continue
            out_pairs.append(out_pair)
    return out_pairs

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

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    if "aecc" in cfg['data']['trainset']:
        aeccdataset = AECCDATASET(**cfg['data']['aecc'])
        print(f"the length of data is {len(aeccdataset)}")

    # if "timit" in cfg['data']['train']:
    #     timit = TIMITDATASET()

if __name__ == "__main__":
    unit_test()