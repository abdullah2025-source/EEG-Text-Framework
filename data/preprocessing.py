import os
import numpy as np
import mne
from scipy.signal import spectrogram, hilbert
import torch
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# EEG Preprocessing
# ----------------------------------------------------------------------

def preprocess_eeg(raw_path, tmin=0, tmax=None, filter_low=0.5, filter_high=50, notch=60, segment_duration=2.0):
    """
    Load raw EEG file, filter, ICA, and segment into fixed-length windows.
    Returns array of shape (n_segments, n_channels, n_times) and labels (if any).
    """
    raw = mne.io.read_raw_edf(raw_path, preload=True)
    raw.filter(filter_low, filter_high, fir_design='firwin')
    raw.notch_filter(notch, fir_design='firwin')
    # ICA artifact removal (optional, simplified)
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw)
    raw = ica.apply(raw)
    # Segment into fixed windows
    events = mne.make_fixed_length_events(raw, duration=segment_duration, overlap=segment_duration/2)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=segment_duration, baseline=None, preload=True)
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    return data

def compute_spectrogram(eeg_segment, fs=250, nperseg=128):
    """Compute spectrogram for each channel. Returns shape (n_channels, n_freq, n_time)."""
    spec = []
    for ch in range(eeg_segment.shape[0]):
        f, t, Sxx = spectrogram(eeg_segment[ch], fs=fs, nperseg=nperseg)
        spec.append(Sxx)
    return np.array(spec)  # (ch, freq, time)

def compute_adjacency(eeg_segment, method='plv'):
    """Compute functional connectivity matrix (n_channels, n_channels) using Phase Locking Value."""
    n_ch = eeg_segment.shape[0]
    adj = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            sig_i = eeg_segment[i]
            sig_j = eeg_segment[j]
            phase_i = np.angle(hilbert(sig_i))
            phase_j = np.angle(hilbert(sig_j))
            plv = np.abs(np.mean(np.exp(1j * (phase_i - phase_j))))
            adj[i, j] = adj[j, i] = plv
    return adj

# ----------------------------------------------------------------------
# Text Preprocessing
# ----------------------------------------------------------------------

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_reddit_data(data_dir, split='train'):
    """
    Load Reddit RSI dataset.
    Expected: data_dir/raw/Reddit/train.csv, test.csv with columns 'text', 'label'.
    """
    file_map = {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}
    df = pd.read_csv(os.path.join(data_dir, 'raw', 'Reddit', file_map[split]))
    texts = df['text'].tolist()
    labels = df['label'].values.astype(int)
    return texts, labels

def load_clpsych_data(data_dir, split='train'):
    """
    Load CLPsych dataset.
    Expected: data_dir/raw/CLPsych/clpsych_2024.json with fields 'post' and 'risk_level'.
    We'll split randomly if no pre-split files.
    """
    with open(os.path.join(data_dir, 'raw', 'CLPsych', 'clpsych_2024.json'), 'r') as f:
        data = json.load(f)
    texts = [item['post'] for item in data]
    labels = [item['risk_level'] for item in data]
    # Simple split (0-3 labels)
    if split == 'train':
        X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        return X_train, np.array(y_train)
    elif split == 'val':
        _, X_val, _, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        # further split into val/test? We'll just use 10% val, 10% test from the 20% holdout
        X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
        return X_val, np.array(y_val)
    else:  # test
        _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
        return X_test, np.array(y_test)

# ----------------------------------------------------------------------
# EEG Dataset Loaders (actual data from TUH/PhysioNet)
# ----------------------------------------------------------------------

def load_tuh_eeg(data_dir, split='train'):
    """
    Load preprocessed TUH EEG segments.
    We assume the data has been preprocessed and stored as .npy files per subject.
    For simplicity, we will use a dummy implementation that returns random data.
    In practice, you would read actual files.
    """
    # Placeholder: replace with actual loading logic.
    # For demonstration, we create random data.
    n_segments = 100 if split == 'train' else 20
    n_channels = 64
    n_times = 500   # 2 seconds at 250 Hz
    segments = [np.random.randn(n_channels, n_times) for _ in range(n_segments)]
    labels = np.random.randint(0, 4, size=n_segments)   # dummy labels (0-3)
    return segments, labels

def load_physionet_mi(data_dir, split='train'):
    """
    Load preprocessed PhysioNet Motor Imagery EEG segments.
    Placeholder similar to above.
    """
    n_segments = 80 if split == 'train' else 20
    n_channels = 64
    n_times = 500
    segments = [np.random.randn(n_channels, n_times) for _ in range(n_segments)]
    labels = np.random.randint(0, 4, size=n_segments)
    return segments, labels

# ----------------------------------------------------------------------
# Pairing Logic
# ----------------------------------------------------------------------

def create_pairs(eeg_labels, text_labels, k=3):
    """
    Given lists of labels for EEG and text, return a list of (eeg_idx, list_of_text_idx) pairs.
    For each EEG index, choose k text indices with the same label.
    """
    from collections import defaultdict
    text_by_class = defaultdict(list)
    for idx, lbl in enumerate(text_labels):
        text_by_class[lbl].append(idx)

    pairs = []
    for eeg_idx, lbl in enumerate(eeg_labels):
        candidates = text_by_class.get(lbl, [])
        if len(candidates) >= k:
            sampled = np.random.choice(candidates, k, replace=False).tolist()
        else:
            sampled = np.random.choice(candidates, k, replace=True).tolist()
        pairs.append((eeg_idx, sampled))
    return pairs
