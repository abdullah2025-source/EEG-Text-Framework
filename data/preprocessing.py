import os
import numpy as np
import mne
import torch
from scipy.signal import spectrogram
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# EEG Preprocessing
# ----------------------------------------------------------------------

def preprocess_eeg(raw_path, tmin=0, tmax=None, filter_low=0.5, filter_high=50, notch=60):
    """
    Load raw EEG, filter, ICA, and segment into windows.
    Returns: array of shape (n_segments, n_channels, n_times)
    """
    raw = mne.io.read_raw_edf(raw_path, preload=True)
    # Apply filters
    raw.filter(filter_low, filter_high, fir_design='firwin')
    raw.notch_filter(notch, fir_design='firwin')
    # ICA for artifact removal (optional, simplified)
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw)
    raw = ica.apply(raw)
    # Segment into windows (e.g., 2–5 s)
    events = mne.make_fixed_length_events(raw, duration=2.0, overlap=0.5)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    return epochs.get_data()  # (n_segments, n_channels, n_times)

def compute_spectrogram(eeg_segment, fs=250, nperseg=128):
    """
    Compute spectrogram for each channel and return as array (n_channels, freq_bins, time_bins)
    """
    spec = []
    for ch in range(eeg_segment.shape[0]):
        f, t, Sxx = spectrogram(eeg_segment[ch], fs=fs, nperseg=nperseg)
        spec.append(Sxx)
    return np.array(spec)  # (n_channels, n_freq, n_time)

def compute_adjacency(eeg_segment, method='plv'):
    """
    Compute functional connectivity matrix (n_channels, n_channels) using Phase Locking Value.
    """
    from scipy.signal import hilbert
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
    """Basic cleaning: lowercasing, remove URLs, etc."""
    import re
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_reddit_data(data_dir):
    """
    Load Reddit RSI dataset.
    Expected structure: data_dir/raw/Reddit/
        train.csv, test.csv with columns: text, label
    Label mapping: 0=no, 1=low, 2=moderate, 3=severe
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(data_dir, 'raw', 'Reddit', 'train.csv'))
    texts = df['text'].apply(clean_text).tolist()
    labels = df['label'].values
    return texts, labels

def load_clpsych_data(data_dir):
    """
    Load CLPsych Shared Task data.
    Expected: data_dir/raw/CLPsych/ with json or csv.
    """
    import json
    with open(os.path.join(data_dir, 'raw', 'CLPsych', 'clpsych_2024.json'), 'r') as f:
        data = json.load(f)
    texts = [item['post'] for item in data]
    labels = [item['risk_level'] for item in data]  # assume 0-3
    return texts, np.array(labels)

def load_tuh_eeg(data_dir):
    """
    Load TUH EEG corpus. This is large; we assume preprocessed segments stored as .npy.
    We'll implement a placeholder.
    """
    # In reality, you would iterate through the TUH directory structure.
    # Here we assume precomputed segments in data_dir/processed/TUH/.
    seg_dir = os.path.join(data_dir, 'processed', 'TUH')
    files = [f for f in os.listdir(seg_dir) if f.endswith('_segment.npy')]
    segments = [np.load(os.path.join(seg_dir, f)) for f in files]
    # Labels: none (weak supervision uses class from text)
    # Return segments and placeholder labels (all 0 for now, will be paired later)
    labels = np.zeros(len(segments), dtype=int)
    return segments, labels

def load_physionet_mi(data_dir):
    """
    Load PhysioNet Motor Imagery EEG.
    Expected preprocessed segments.
    """
    seg_dir = os.path.join(data_dir, 'processed', 'PhysioNet')
    files = [f for f in os.listdir(seg_dir) if f.endswith('_segment.npy')]
    segments = [np.load(os.path.join(seg_dir, f)) for f in files]
    # Labels: task labels (but we will use risk classes from pairing)
    labels = np.zeros(len(segments), dtype=int)
    return segments, labels

# ----------------------------------------------------------------------
# Pairing Helper
# ----------------------------------------------------------------------

def create_pairs(eeg_segments, eeg_labels, text_samples, text_labels, k=3):
    """
    Create EEG-text pairs for contrastive learning.
    For each EEG segment, randomly sample k text samples of the same class.
    Returns list of tuples: (eeg_idx, text_idx_list)
    """
    from collections import defaultdict
    text_by_class = defaultdict(list)
    for idx, lbl in enumerate(text_labels):
        text_by_class[lbl].append(idx)

    pairs = []
    for eeg_idx, lbl in enumerate(eeg_labels):
        candidates = text_by_class[lbl]
        if len(candidates) >= k:
            sampled = np.random.choice(candidates, k, replace=False).tolist()
        else:
            sampled = np.random.choice(candidates, k, replace=True).tolist()
        pairs.append((eeg_idx, sampled))
    return pairs
