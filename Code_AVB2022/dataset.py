import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
import augment
from data_augment import ChainRunner, random_time_warp, random_pitch_shift
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
from functools import partial
import hydra


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.wav_path = cfg.dataset.wav_path
        self.feat_path = cfg.dataset.feat_path
        self.features = cfg.dataset.features

        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size

        ds = hydra.utils.instantiate(self.cfg.dataset.dataset, phase, self.cfg)
        # ds = ExvoDataset(phase, self.cfg)
        dl = DataLoader(
            ds,
            batch_size,
            phase_cfg.shuffle,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=ds.collate_fn,
        )

        return dl

    def train_dataloader(self):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")


class DADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wav_path = cfg.dataset.wav_path
        self.feat_path = cfg.dataset.feat_path
        self.features = cfg.dataset.features

        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        # ds = hydra.utils.instantiate(self.cfg.dataset.dataset, phase, self.cfg)
        if phase == "train":
            ds = ExvoSpeakerDataset(phase, self.cfg)
        else:
            ds = ExvoDataset(phase, self.cfg)
        dl = DataLoader(
            ds,
            batch_size,
            phase_cfg.shuffle,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=ds.collate_fn,
        )

        return dl

    def train_dataloader(self):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")


class ExvoDataset(Dataset):
    emotion_labels = [
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]
    emotion2index = {e: i for i, e in enumerate(emotion_labels)}

    all_features = ["compare", "deepspectrum", "egemaps", "openxbow"]

    def __init__(self, phase, cfg):
        phase_cfg = cfg.dataset.get(phase)
        ocwd = hydra.utils.get_original_cwd()
        self.cfg = cfg
        self.phase = phase
        self.csv_path = join(ocwd, phase_cfg.csv_path)
        self.wav_path = join(ocwd, cfg.dataset.wav_path)
        self.feat_path = join(ocwd, cfg.dataset.feat_path)
        self.sr = cfg.dataset.sr
        self.wav = cfg.dataset.wav

        self.features = cfg.dataset.features

        self.csv = self.read_csv(self.csv_path)

        self.emotion_label_order = self.get_emotion_label_order()

        if cfg.dataset.augment.enable:
            chain = augment.EffectChain()
            chain.pitch(
                partial(
                    random_pitch_shift,
                    a=0 - cfg.dataset.augment.pitch,
                    b=cfg.dataset.augment.pitch,
                )
            ).rate(16000)
            chain.tempo(partial(random_time_warp, f=cfg.dataset.augment.rate))
            self.chain = ChainRunner(chain)

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df

    def get_emotion_label_order(self):
        mode = self.cfg.dataset.emotion_label_order
        # high performance to low
        h2l_order = [
            "Awe",
            "Surprise",
            "Amusement",
            "Fear",
            "Horror",
            "Sadness",
            "Distress",
            "Excitement",
            "Triumph",
            "Awkwardness",
        ]

        # frquent to rare
        f2r_order = [
            "Amusement",
            "Surprise",
            "Fear",
            "Sadness",
            "Distress",
            "Excitement",
            "Awe",
            "Horror",
            "Awkwardness",
            "Triumph",
        ]

        if mode == "default":
            out = self.emotion_labels  # default order used by the data
        elif mode == "h2l":
            out = h2l_order
        elif mode == "l2h":
            out = list(reversed(h2l_order))
        elif mode == "f2r":
            out = f2r_order
        elif mode == "r2f":
            out = list(reversed(f2r_order))
        else:
            raise ValueError(f"Unknown emotion label order mode: {mode}")
        return out

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        out = {}
        wav_id = self.csv.loc[index, "id"]
        fid = wav_id[:-4]
        out["fid"] = fid
        speaker = int(self.csv.loc[index, "speaker"].split("_")[-1])
        main_emotion = self.emotion2index[self.csv.loc[index, "type"]]
        emotion = (
            self.csv.loc[index, self.emotion_label_order].to_numpy().astype("float")
        )

        if self.wav:
            wav = self.load_wav(fid)
            out["wav"] = wav

        features = self.load_features(fid)
        speaker = torch.LongTensor([speaker])
        emotion = torch.from_numpy(emotion)
        main_emotion = torch.LongTensor([main_emotion])

        out["features"] = features
        out["speaker"] = speaker
        out["emotion"] = emotion
        out["main_emotion"] = main_emotion

        return out

    def collate_fn(self, batch):
        fids = [b["fid"] for b in batch]
        speaker = torch.stack([b["speaker"] for b in batch]).squeeze()
        emotion = torch.stack([b["emotion"] for b in batch])
        main_emotion = torch.stack([b["main_emotion"] for b in batch]).squeeze()

        feat_names = list(batch[0]["features"].keys())
        out = {
            "fids": fids,
            "speaker": speaker,
            "emotion": emotion,
            "main_emotion": main_emotion,
        }
        features = []
        for feat_name in feat_names:
            feat = [b["features"][feat_name] for b in batch]
            feat = torch.stack(feat)
            features.append(feat)
            out[feat_name] = feat
        feature = torch.cat(features, dim=-1)
        out["feature"] = feature

        if self.cfg.dataset.wav:
            max_length = max(b["wav"].shape[1] for b in batch)
            buf = [F.pad(b["wav"], [0, max_length - b["wav"].shape[1]]) for b in batch]
            wav = torch.stack(buf).squeeze(1)
            out["wav"] = wav
        return out

    def load_wav(self, fid):
        path = join(self.wav_path, f"{fid}.wav")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        assert sr == self.sr
        if self.phase == "train" and self.cfg.dataset.augment.enable:
            wav = self.chain(wav)

        max_length = int(self.cfg.dataset.max_wav_length * sr)

        # import pdb; pdb.set_trace()
        # copy
        # if self.cfg.dataset.padding:
        #     pass

        if self.cfg.dataset.copy:
            while wav.shape[1] < max_length:
                silence_padded = torch.zeros(1, max_length - wav.shape[1])
                wav = torch.cat([silence_padded, wav], -1)

        # # copy
        # if self.cfg.dataset.copy: # -> False
        #     while wav.shape[1] < max_length:
        #         # print('before:', wav.shape)
        #         wav = torch.cat([wav, wav], -1)
        #         # print('after', wav.shape)

        # truncate
        if wav.shape[1] > max_length:
            # idx = random.randint(0, wav.shape[1] - max_length)
            idx = (wav.shape[1] - max_length) + 1
            wav = wav[:, idx : idx + max_length]
            # print('final: ', wav.shape)
        return wav

    def load_features(self, fid):
        out = {}
        for feat_name in self.features:
            feat = getattr(self, f"get_{feat_name}")(fid)
            out[feat_name] = feat
        return out

    def get_compare(self, fid):
        feat_path = join(self.feat_path, "compare", f"{fid}.csv")
        with open(feat_path, "r", encoding="utf8") as f:
            lines = [line.strip().split(",") for line in f if line.strip()]
        feat = [float(item) for item in lines[1][1:]]
        feat = torch.FloatTensor(feat)
        return feat

    def get_egemaps(self, fid):
        # DEBUG
        feat_path = join(self.feat_path, "egemaps", f"{fid}.csv")
        with open(feat_path, "r", encoding="utf8") as f:
            lines = [line.strip().split(",") for line in f if line.strip()]
        feat = [float(item) for item in lines[1][1:]]
        feat = torch.FloatTensor(feat)
        return feat


class ExvoSpeakerDataset(ExvoDataset):
    def __init__(self, phase, cfg):
        super().__init__(phase, cfg)
        self.spkr2data = {
            item[0]: item[1] for item in list(self.csv.groupby("speaker"))
        }
        self.spkrs = list(self.spkr2data)
        self.spkrs.sort()
        self.target_speaker = None

    def get_speakers(self):
        return self.spkrs.copy()

    def get_target_speaker(self):
        return self.target_speaker

    def set_target_speaker(self, spkr):
        self.target_speaker = spkr

    def __len__(self):
        return len(self.spkr2data[self.target_speaker])

    def __getitem__(self, index):
        out = {}
        df = self.spkr2data[self.target_speaker]
        wav_id = df.loc[index, "id"]
        fid = wav_id[:-4]
        out["fid"] = fid
        speaker = int(df.loc[index, "speaker"].split("_")[-1])
        main_emotion = self.emotion2index[df.loc[index, "type"]]
        emotion = df.loc[index, self.emotion_label_order].to_numpy().astype("float")

        if self.wav:
            wav = self.load_wav(fid)
            out["wav"] = wav

        features = self.load_features(fid)
        speaker = torch.LongTensor([speaker])
        emotion = torch.from_numpy(emotion)
        main_emotion = torch.LongTensor([main_emotion])

        out["features"] = features
        out["speaker"] = speaker
        out["emotion"] = emotion
        out["main_emotion"] = main_emotion

        return out


import transformers


class ExvoDataset_Spec(Dataset):
    emotion_labels = [
        "Awe",
        "Excitement",
        "Amusement",
        "Awkwardness",
        "Fear",
        "Horror",
        "Distress",
        "Triumph",
        "Sadness",
        "Surprise",
    ]
    emotion2index = {e: i for i, e in enumerate(emotion_labels)}

    all_features = ["compare", "deepspectrum", "egemaps", "openxbow"]

    def __init__(self, phase, cfg):
        phase_cfg = cfg.dataset.get(phase)
        ocwd = hydra.utils.get_original_cwd()
        self.cfg = cfg
        self.phase = phase
        self.csv_path = join(ocwd, phase_cfg.csv_path)
        self.wav_path = join(ocwd, cfg.dataset.wav_path)
        self.feat_path = join(ocwd, cfg.dataset.feat_path)
        self.sr = cfg.dataset.sr
        self.wav = cfg.dataset.wav

        self.features = cfg.dataset.features

        self.csv = self.read_csv(self.csv_path)

        self.emotion_label_order = self.get_emotion_label_order()

        if cfg.dataset.augment.enable:
            chain = augment.EffectChain()
            chain.pitch(
                partial(
                    random_pitch_shift,
                    a=0 - cfg.dataset.augment.pitch,
                    b=cfg.dataset.augment.pitch,
                )
            ).rate(16000)
            chain.tempo(partial(random_time_warp, f=cfg.dataset.augment.rate))
            self.chain = ChainRunner(chain)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            n_fft=1024, hop_length=512, center=True, power=2.0
        )

        self.dino_feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(
            cfg.model.ssl_model_1, cache_dir="model_tmp"  # "facebook/dino-vitb8"
        )

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df

    def get_emotion_label_order(self):
        mode = self.cfg.dataset.emotion_label_order
        # high performance to low
        h2l_order = [
            "Awe",
            "Surprise",
            "Amusement",
            "Fear",
            "Horror",
            "Sadness",
            "Distress",
            "Excitement",
            "Triumph",
            "Awkwardness",
        ]

        # frquent to rare
        f2r_order = [
            "Amusement",
            "Surprise",
            "Fear",
            "Sadness",
            "Distress",
            "Excitement",
            "Awe",
            "Horror",
            "Awkwardness",
            "Triumph",
        ]

        if mode == "default":
            out = self.emotion_labels  # default order used by the data
        elif mode == "h2l":
            out = h2l_order
        elif mode == "l2h":
            out = list(reversed(h2l_order))
        elif mode == "f2r":
            out = f2r_order
        elif mode == "r2f":
            out = list(reversed(f2r_order))
        else:
            raise ValueError(f"Unknown emotion label order mode: {mode}")
        return out

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        out = {}
        wav_id = self.csv.loc[index, "id"]
        fid = wav_id[:-4]
        out["fid"] = fid
        speaker = int(self.csv.loc[index, "speaker"].split("_")[-1])
        main_emotion = self.emotion2index[self.csv.loc[index, "type"]]
        emotion = (
            self.csv.loc[index, self.emotion_label_order].to_numpy().astype("float")
        )

        if self.wav:
            wav = self.load_wav(fid)
            mel_spec = self.load_mel_spec(wav)

            out["wav"] = wav
            out["mel_spec"] = mel_spec

        features = self.load_features(fid)
        speaker = torch.LongTensor([speaker])
        emotion = torch.from_numpy(emotion)
        main_emotion = torch.LongTensor([main_emotion])

        out["features"] = features
        out["speaker"] = speaker
        out["emotion"] = emotion
        out["main_emotion"] = main_emotion

        return out

    def collate_fn(self, batch):
        fids = [b["fid"] for b in batch]
        speaker = torch.stack([b["speaker"] for b in batch]).squeeze()
        emotion = torch.stack([b["emotion"] for b in batch])
        main_emotion = torch.stack([b["main_emotion"] for b in batch]).squeeze()

        feat_names = list(batch[0]["features"].keys())
        out = {
            "fids": fids,
            "speaker": speaker,
            "emotion": emotion,
            "main_emotion": main_emotion,
        }
        features = []
        for feat_name in feat_names:
            feat = [b["features"][feat_name] for b in batch]
            feat = torch.stack(feat)
            features.append(feat)
            out[feat_name] = feat
        feature = torch.cat(features, dim=-1)
        out["feature"] = feature

        mel_spec = torch.cat([b["mel_spec"] for b in batch], dim=0)
        out["mel_spec"] = mel_spec

        if self.cfg.dataset.wav:
            max_length = max(b["wav"].shape[1] for b in batch)
            buf = [F.pad(b["wav"], [0, max_length - b["wav"].shape[1]]) for b in batch]
            wav = torch.stack(buf).squeeze(1)
            out["wav"] = wav
        return out

    def load_wav(self, fid):
        path = join(self.wav_path, f"{fid}.wav")
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        assert sr == self.sr
        if self.phase == "train" and self.cfg.dataset.augment.enable:
            wav = self.chain(wav)

        max_length = int(self.cfg.dataset.max_wav_length * sr)

        # import pdb; pdb.set_trace()
        # copy
        # if self.cfg.dataset.padding:
        #     pass

        if self.cfg.dataset.copy:
            while wav.shape[1] < max_length:
                silence_padded = torch.zeros(1, max_length - wav.shape[1])
                wav = torch.cat([silence_padded, wav], -1)

        # # copy
        # if self.cfg.dataset.copy: # -> False
        #     while wav.shape[1] < max_length:
        #         # print('before:', wav.shape)
        #         wav = torch.cat([wav, wav], -1)
        #         # print('after', wav.shape)

        # truncate
        if wav.shape[1] > max_length:
            # idx = random.randint(0, wav.shape[1] - max_length)
            idx = (wav.shape[1] - max_length) + 1
            wav = wav[:, idx : idx + max_length]
            # print('final: ', wav.shape)

        # mel_transform = torchaudio.transforms.MelSpectrogram(n_fft = 1024, hop_length=512, center=True, power=2.0)
        # mel_spec = mel_transform(wav)

        return wav

    def load_mel_spec(self, wav):
        mel_spec = self.mel_transform(wav)
        mel_spec = mel_spec.squeeze(0)

        # 1 channel -> 3 channels
        mel_spec = torch.stack([mel_spec, mel_spec, mel_spec])

        # Use DINO Image preprocessing functions
        mel_spec = self.dino_feature_extractor(images=[mel_spec], return_tensors='pt')['pixel_values']

        return mel_spec

    def load_features(self, fid):
        out = {}
        for feat_name in self.features:
            feat = getattr(self, f"get_{feat_name}")(fid)
            out[feat_name] = feat
        return out

    def get_compare(self, fid):
        feat_path = join(self.feat_path, "compare", f"{fid}.csv")
        with open(feat_path, "r", encoding="utf8") as f:
            lines = [line.strip().split(",") for line in f if line.strip()]
        feat = [float(item) for item in lines[1][1:]]
        feat = torch.FloatTensor(feat)
        return feat

    def get_egemaps(self, fid):
        # DEBUG
        feat_path = join(self.feat_path, "egemaps", f"{fid}.csv")
        with open(feat_path, "r", encoding="utf8") as f:
            lines = [line.strip().split(",") for line in f if line.strip()]
        feat = [float(item) for item in lines[1][1:]]
        feat = torch.FloatTensor(feat)
        return feat


if __name__ == "__main__":
    # For testing
    dataset = ExvoDataset(
        # "filelists/exvo_train.csv",
        "data/labels/filelists/exvo_train.csv",
        "../data/wav",
        "../data/feats",
        ["egemaps"],
        16000,
    )
    loader = DataLoader(dataset, 8, collate_fn=dataset.collate_fn)
    print(len(loader))
    for batch in loader:
        print(batch)
        break
