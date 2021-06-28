import os
import logging
import json
import csv
from collections import OrderedDict

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
import torchaudio as ta
import numpy as np
import torch.utils.data as tdata
import timit_utils as tu

from raaec.DSP.torch_DSP import lengths_sub
from raaec.utils.set_config import hydra_runner
from raaec.data.align import correlation_index
from raaec.module.agc import FIXGAIN

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

class AECC_SYNTHETIC_DATASET(tdata.Dataset):
    def __init__(
            self,
            path,
            split='train',
            is_farend_noisy=[0, 1],
            is_nearend_noisy=[0,1],
            fs = 16000,
            scale=False,
            check=False
        ):
        super().__init__()
        meta_path = os.path.join(path, 'meta.csv')
        assert os.path.isfile(meta_path), f"{meta_path} not exists"
        assert split in ['train', 'test']

        self.scale = scale
        self.fs = fs
        self.audio_pairs = []
        meta_csv = csv.DictReader(open(meta_path, "r"))
        for i, row in enumerate(meta_csv):
            audio_pair = {}
            if row['split'] != split:
                continue
            if int(row['is_farend_noisy']) not in is_farend_noisy:
                continue
            if int(row['is_nearend_noisy']) not in is_nearend_noisy:
                continue
            audio_pair['ref'] = os.path.join(
                path, 'farend_speech', f"farend_speech_fileid_{row['fileid']}.wav")
            audio_pair['rec'] = os.path.join(
                path, 'nearend_mic_signal', f"nearend_mic_fileid_{row['fileid']}.wav")
            audio_pair['near'] = os.path.join(
                path, 'nearend_speech', f"nearend_speech_fileid_{row['fileid']}.wav")
            for v in audio_pair.values():
                assert os.path.isfile(v), f"{v} file donesn't exists"
            nearend_scale = float(row['nearend_scale'])
            assert type(nearend_scale) == float, \
                f"the {i}th row's nearend_scale of {meta_path} is not float, is {row['nearend_scale']}"
            audio_pair['near_scale'] = nearend_scale
            self.audio_pairs.append(audio_pair)
        if check:
            self.check()
            logging.info(f"check done with path {path}")

    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        logging.debug(f"reading data from {audio_pair}")
        signals = OrderedDict()
        for signal_name in ('ref', 'rec', 'near'):
            signal, fs = ta.load(audio_pair[signal_name], normalize=True)
            if fs != self.fs:
                logging.logging.warn(
                    f""""the audio {audio_pair['signal_name']}'s sample frequency
                    is not equal the system sample frequencey {self.fs}""")
                signal = ta.functional.resample(signal, fs, self.fs)
            signals[signal_name] = signal.squeeze(0)
        if self.scale:
            signals['near'] *= audio_pair['near_scale']
        return list(signals.values())

    def check(self):
        logging.debug(f"checking data")
        for i in tqdm(range(self.__len__())):
            try:
                self.__getitem__(i)
            except Exception as e:
                print(e)
                print(f"{self.audio_pairs[i]}")
                exit(5)

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
            align=False,
            check=False,
            align_device=None,
    ):
        super().__init__()
        self.audio_pairs = self.build(path, os.listdir(path)) \
            if os.path.isdir(path) else self.load(path)
        
        self.fs = fs
        if align:
            self.gain = FIXGAIN()
            align_device = torch.device(align_device)
            self.align(align_device)
        if dump_path is not None:
            self.dump(dump_path)
        if check:
            self.check()
            logging.info(f"check done with path {path}")
        self.select(select)

    def __getitem__(self, index):
        audio_pair = self.audio_pairs[index]
        logging.debug(f"reading data from {audio_pair}")
        ref, _ = ta.load(audio_pair['ref'], normalize=True)
        rec, _ = ta.load(audio_pair['rec'], normalize=True)
        near, _ = ta.load(audio_pair['near'], normalize=True)
        return ref.squeeze(0), rec.squeeze(0), near.squeeze(0)

    def __len__(self):
        return len(self.audio_pairs)

    def dump(self, dump_path):
        with open(dump_path, "w") as fw:
            for audio_pair in self.audio_pairs:
                fw.write(f"{json.dumps(audio_pair)}\n")
        logging.info(f"dump data in {dump_path} with length {len(self)}")

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

    def build(self, dir, audio_list):
        audio_items = []
        for file in audio_list:
            if os.path.splitext(file)[1] not in [".mp3", ",flac", ".wav"]:
                continue
            for str_find in ['farend', 'doubletalk', 'sweep']:
                str_idx = file.find(str_find)
                if str_idx != -1:
                    id = file[0:str_idx-1]
                    audio_type = file[str_idx:].rsplit('_', 1)[0]
                    audio_items.append([id, audio_type])
                    break

        audio_pairs = {}
        for audio_item in audio_items:
            near_path = os.path.join(dir, f"{audio_item[0]}_nearend_singletalk_mic.wav")
            if not os.path.isfile(near_path):
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
                    "near": f"{prefix1}_nearend_singletalk_mic.wav",
                    'type': v,
                }
                if not (os.path.isfile(out_pair['ref']) and os.path.isfile(out_pair['rec'])):
                    logging.warning(f"{out_pair['ref']} or {out_pair['rec']} is not exist, skip")
                    continue
                out_pairs.append(out_pair)
        return out_pairs

    def check(self):
        logging.debug(f"checking data")
        for i in tqdm(range(self.__len__())):
            try:
                self.__getitem__(i)
            except Exception as e:
                print(e)
                print(f"{self.audio_pairs[i]}")
                exit(5)

    def select(self, select):
        audio_pairs = []
        for item in self.audio_pairs:
            if item['type'] in select:
                audio_pairs.append(item)
        self.audio_pairs = audio_pairs
    
    def align(self, device):
        audio_pairs = []
        with torch.no_grad():
            for audio_pair in tqdm(self.audio_pairs):
                logging.debug(f"reading data from {audio_pair}")
                ref, _ = ta.load(audio_pair['ref'], normalize=False)
                rec, _ = ta.load(audio_pair['rec'], normalize=False)
                near, _ = ta.load(audio_pair['near'], normalize=False)
                signals = [ref, rec, near]
                signals = [self.gain(x).to(device) for x in signals]
                min_len, time_idxs = correlation_index(signals)
                audio_pair['time_idxs'] = time_idxs
                audio_pair['min_len'] = min_len
                audio_pairs.append(audio_pair)
            self.audio_pairs = audio_pairs

@hydra_runner(config_path=os.path.join(os.getcwd(), "conf"), config_name="test")
def unit_test(cfg: DictConfig):
    logging.basicConfig(
        level=eval(f"logging.{cfg['logging'].get('level', 'INFO')}"),
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    train_datasets = []
    for dataset_conf in cfg['data']['dataset']['train']:
        dataset_class = eval(f"{dataset_conf['select']}")
        dataset = dataset_class(**dataset_conf['conf'])
        train_datasets.append(dataset)

    val_datasets = []
    for dataset_conf in cfg['data']['dataset']['val']:
        dataset_class = eval(f"{dataset_conf['select']}")
        dataset = dataset_class(**dataset_conf['conf'])
        val_datasets.append(dataset)

    test_datasets = []
    for dataset_conf in cfg['data']['dataset']['test']:
        dataset_class = eval(f"{dataset_conf['select']}")
        dataset = dataset_class(**dataset_conf['conf'])
        test_datasets.append(dataset)
    

if __name__ == "__main__":
    unit_test()