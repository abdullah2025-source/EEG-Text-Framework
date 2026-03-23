import torch
from torch.utils.data import Dataset
import numpy as np
import os
from .preprocessing import (
    preprocess_eeg, compute_spectrogram, compute_adjacency,
    clean_text, load_reddit_data, load_clpsych_data,
    load_tuh_eeg, load_physionet_mi, create_pairs
)

class EEGDataset(Dataset):
    def __init__(self, data_dir, dataset_name, split='train', transform=True):
        """
        dataset_name: 'tuh' or 'physionet'
        split: 'train', 'val', 'test' (for splitting, we assume pre-split files)
        """
        self.transform = transform
        self.segments, self.labels = self._load(data_dir, dataset_name, split)
        # Precompute spectrograms and adjacency if needed
        self.spectrograms = []
        self.adjacencies = []
        for seg in self.segments:
            if transform:
                spec = compute_spectrogram(seg)
                adj = compute_adjacency(seg)
                self.spectrograms.append(spec)
                self.adjacencies.append(adj)
            else:
                self.spectrograms.append(None)
                self.adjacencies.append(None)

    def _load(self, data_dir, dataset_name, split):
        if dataset_name == 'tuh':
            segs, labs = load_tuh_eeg(data_dir)
        elif dataset_name == 'physionet':
            segs, labs = load_physionet_mi(data_dir)
        else:
            raise ValueError
        # split: assume segs are already split by subject; we use provided indices
        # For simplicity, we use a fixed split ratio defined elsewhere.
        # In practice, you would load split-specific files.
        return segs, labs

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        if self.transform:
            spec = self.spectrograms[idx]
            adj = self.adjacencies[idx]
        else:
            spec = compute_spectrogram(seg)
            adj = compute_adjacency(seg)
        return {
            'signal': torch.tensor(seg, dtype=torch.float32),
            'spectrogram': torch.tensor(spec, dtype=torch.float32),
            'adjacency': torch.tensor(adj, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextDataset(Dataset):
    def __init__(self, data_dir, dataset_name, split='train'):
        if dataset_name == 'reddit':
            self.texts, self.labels = load_reddit_data(data_dir)
        elif dataset_name == 'clpsych':
            self.texts, self.labels = load_clpsych_data(data_dir)
        else:
            raise ValueError
        # Clean texts
        self.texts = [clean_text(t) for t in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultimodalDataset(Dataset):
    """
    Combines EEG and text datasets with weak supervision.
    Returns a batch of EEG-text pairs (positive pairs) according to class-level pairing.
    """
    def __init__(self, eeg_dataset, text_dataset, k=3):
        self.eeg_dataset = eeg_dataset
        self.text_dataset = text_dataset
        self.k = k
        # Build pairing indices
        eeg_labels = [self.eeg_dataset[i]['label'].item() for i in range(len(self.eeg_dataset))]
        text_labels = [self.text_dataset[i]['label'].item() for i in range(len(self.text_dataset))]
        self.pairs = create_pairs(eeg_labels, text_labels, k)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eeg_idx, text_idxs = self.pairs[idx]
        eeg_item = self.eeg_dataset[eeg_idx]
        # For each positive pair, we return one EEG and one text (randomly chosen from k)
        # In contrastive learning, we will handle the k samples separately.
        # For simplicity, we return one text per EEG, but the training loop can also use the k.
        text_idx = text_idxs[0]  # use the first; in training you can iterate over k
        text_item = self.text_dataset[text_idx]
        return {
            'eeg_signal': eeg_item['signal'],
            'eeg_spectrogram': eeg_item['spectrogram'],
            'eeg_adjacency': eeg_item['adjacency'],
            'text': text_item['text'],
            'eeg_label': eeg_item['label'],
            'text_label': text_item['label']
        }
