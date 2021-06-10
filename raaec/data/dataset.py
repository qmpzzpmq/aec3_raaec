import os
import logging
import json

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
            self, dirpath, subset, fs=16000, wav_norm=False):
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

class TIMIT_FLTER_AECDATASET(tdata.Dataset):
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

class AECC_REAL_DATASET(tdata.Dataset):
    def __init__(
            self, path,
            select=[
                'farend_singletalk',
                'farend_singletalk_with_movement',
                'doubletalk',
                'doubletalk_with_movement',
                ],
            fs=16000,
            dump_path=None,
    ):
        super().__init__()
        self.audio_pairs = self.build(path, os.listdir(path), select) \
            if os.path.isdir(path) else self.load(path)
        if dump_path is not None:
            self.dump(dump_path)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        ref, _ = ta.load(audio_pair['ref'], normalize=False)
        rec, _ = ta.load(audio_pair['rec'], normalize=False)
        near, _ = ta.load(audio_pair['rec'], normalize=False)
        return ref.squeeze(0), rec.squeeze(0), near.squeeze(0)

    def __len__(self):
        return len(self.audio_pairs)

    def dump(self, dump_path):
        with open(dump_path, "w") as fw:
            for audio_pair in self.audio_pairs:
                fw.write(f"{json.dumps(audio_pair)}\n")

    def load(self, load_path):
        logging.warn(f"using {load_path} file directly for aec challenge data")
        audio_pairs = []
        for i, line in enumerate(open(load_path, 'r')):
            audio_pair = json.loads(line.strip())
            if len(audio_pair) != 3:
                logging.warn(f"the length data in {i}th line of {load_path} is not 3")
                continue
            audio_pairs.append(audio_pair)
        return audio_pairs

    def build(self, dir, audio_list, select):
        audio_items = [x.split("_")[0:-1] for x in audio_list \
            if os.path.splitext(x)[1] in [".mp3", ",flac", ".wav"]]
        for i, x in enumerate(audio_items):
            assert len(x) > 1, f"x: {x}, i: {i}, item: {audio_items[i]}, list: {audio_list[i]}"
        audio_items = [[x[0], "_".join(x[1:])] for x in audio_items]
        audio_pairs = {}
        for audio_item in audio_items:
            near_path = os.path.join(dir, f"{audio_item[0]}_nearend_singletalk_mic.wav")
            if not os.path.isfile(near_path):
                continue
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
                out_pair = {
                    "ref": f"{prefix2}_lpb.wav",
                    "rec": f"{prefix2}_mic.wav",
                    "near": near_path,
                    }
                if not (os.path.isfile(out_pair['ref']) and os.path.isfile(out_pair['rec'])):
                    logging.warning(f"{out_pair['ref']} or {out_pair['rec']} is not exist, skip")
                    continue
                out_pairs.append(out_pair)
        return out_pairs

# class AECC_SYNTHETIC_DATASET(tdata.dataset):
#     def __init__(self, csv_path):
#         super().__init__()
#         assert os.path.isfile(csv_path)
#         base_dir = os.path.dirname(csv_path)
# 
# 
#     def load(self, csv_path, base_dir):

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    train_datasets = []
    for dataset in cfg['data']['dataset']['train']:
        dataset_class = eval(f"{dataset['select']}")
        train_datasets.append(
            dataset_class(**dataset['conf'])
        )

    val_datasets = []
    for dataset in cfg['data']['dataset']['val']:
        dataset_class = eval(f"{dataset['select']}")
        val_datasets.append(
            dataset_class(**dataset['conf'])
        )

    test_datasets = []
    for dataset in cfg['data']['dataset']['test']:
        dataset_class = eval(f"{dataset['select']}")
        test_datasets.append(
            dataset_class(**dataset['conf'])
        )    
    

if __name__ == "__main__":
    unit_test()