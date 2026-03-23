import os
import torch
from torch.utils.data import Dataset
import numpy as np
from .preprocessing import (
    load_tuh_eeg, load_physionet_mi, load_reddit_data, load_clpsych_data,
    create_pairs, compute_spectrogram, compute_adjacency, clean_text
)

class EEGDataset(Dataset):
    """EEG dataset with preprocessed signals, spectrograms, and adjacency matrices."""
    def __init__(self, data_dir, dataset_name, split='train', transform=True):
        """
        Args:
            data_dir: root data directory (e.g., 'data/raw')
            dataset_name: 'tuh' or 'physionet'
            split: 'train', 'val', 'test' (assumes files are already split)
            transform: if True, precompute spectrogram/adjacency (else compute on the fly)
        """
        self.transform = transform
        self.segments, self.labels = self._load(data_dir, dataset_name, split)
        self.spectrograms = []
        self.adjacencies = []
        for seg in self.segments:
            if transform:
                spec = compute_spectrogram(seg)
                adj = compute_adjacency(seg)
                self.spectrograms.append(spec)
                self.adjacencies.append(adj)

    def _load(self, data_dir, dataset_name, split):
        if dataset_name == 'tuh':
            segs, labs = load_tuh_eeg(data_dir, split)
        elif dataset_name == 'physionet':
            segs, labs = load_physionet_mi(data_dir, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
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
    """Text dataset with cleaned text and labels."""
    def __init__(self, data_dir, dataset_name, split='train'):
        if dataset_name == 'reddit':
            self.texts, self.labels = load_reddit_data(data_dir, split)
        elif dataset_name == 'clpsych':
            self.texts, self.labels = load_clpsych_data(data_dir, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
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
    Weakly supervised dataset that pairs EEG segments with text samples of the same class.
    For each EEG segment, k text samples are paired as positive samples.
    """
    def __init__(self, eeg_dataset, text_dataset, k=3):
        self.eeg_dataset = eeg_dataset
        self.text_dataset = text_dataset
        self.k = k
        # Pre-extract labels
        eeg_labels = [self.eeg_dataset[i]['label'].item() for i in range(len(self.eeg_dataset))]
        text_labels = [self.text_dataset[i]['label'].item() for i in range(len(self.text_dataset))]
        self.pairs = create_pairs(eeg_labels, text_labels, k)   # list of (eeg_idx, list_of_text_idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eeg_idx, text_idxs = self.pairs[idx]
        eeg_item = self.eeg_dataset[eeg_idx]
        # For simplicity, return one text from the k positives (the first one)
        # In contrastive learning, we can use the k texts in the loss by expanding.
        text_idx = text_idxs[0]
        text_item = self.text_dataset[text_idx]
        return {
            'eeg_signal': eeg_item['signal'],
            'eeg_spectrogram': eeg_item['spectrogram'],
            'eeg_adjacency': eeg_item['adjacency'],
            'text': text_item['text'],
            'label': eeg_item['label'],   # same as text_item['label']
            'modality_label': torch.tensor(0, dtype=torch.long)  # placeholder, not used directly
        }
