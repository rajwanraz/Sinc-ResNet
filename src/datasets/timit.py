import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class TimitTrain(Dataset):
    def __init__(self, path, chunk_len):
        self.chunk_len = chunk_len
        self.path = path
        self.labels = np.load(os.path.join(path, 'labels.npy'), allow_pickle=True).item()
        with open(os.path.join(path, 'train.scp')) as f:
            self.train_samples = f.read().splitlines()

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        wav_path = self.train_samples[idx]
        label = self.labels[wav_path]
        wav, sr = torchaudio.load(os.path.join(self.path, wav_path))
        start = np.random.randint(len(wav[0]) - self.chunk_len)
        chunk = wav[0, start:start + self.chunk_len]
        volume_gain = 0.4 * torch.randn(1) + 0.8
        return torch.unsqueeze(chunk, 0) * volume_gain, label


class TimitEval(Dataset):
    def __init__(self, path: str, chunk_len: int, chunk_shift: int, split: str):
        self.cur_wav_id, self.cur_chunk = 0, 0
        self.path, self.chunk_len, self.chunk_shift = path, chunk_len, chunk_shift
        self.labels = np.load(os.path.join(path, 'labels.npy'), allow_pickle=True).item()
        with open(os.path.join(path, split)) as f:
            self.test_samples = f.read().splitlines()
        self.wav_path = os.path.join(self.path, self.test_samples[self.cur_wav_id])
        self.cur_wav, _ = torchaudio.load(self.wav_path)

    def __len__(self):
        return len(self.test_samples)

    def __getitem__(self, idx):
        wav_path = self.test_samples[idx]
        label = self.labels.get(wav_path, None)
        if label is None:
          label = wav_path.split('/')[-2]
        wav, sr = torchaudio.load(os.path.join(self.path, wav_path))
        chunks = []
        for start in range(0, len(wav[0]) - self.chunk_len + 1, self.chunk_shift):
            chunks.append(wav[0, start:start + self.chunk_len])
        return torch.unsqueeze(torch.stack(chunks), dim=1), label, len(chunks)
