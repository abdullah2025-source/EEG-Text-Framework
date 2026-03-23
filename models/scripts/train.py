import torch
from torch.utils.data import DataLoader
from models.eeg_encoder import EEGEncoder
from models.text_encoder import TextEncoder
from models.alignment import CLEP, DomainAdversarial
from models.fusion import HypernetworkFusion
from models.classifier import ClassifierHead
from training.losses import CombinedLoss
from training.metrics import compute_metrics
from data.dataset import MultimodalDataset

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        eeg_signal, eeg_spec, adj, text, bin_target, ord_target, modality_label = batch
        eeg_signal = eeg_signal.to(device)
        eeg_spec = eeg_spec.to(device)
        adj = adj.to(device)
        text = text.to(device)
        bin_target = bin_target.to(device)
        ord_target = ord_target.to(device)
        modality_label = modality_label.to(device)

        # Forward
        z_eeg = eeg_encoder(eeg_signal, eeg_spec, adj)
        z_text = text_encoder(text)
        contrast_loss = clep(z_eeg, z_text)
        adv_loss = dann(z_eeg, modality_label) + dann(z_text, modality_label)
        # Fusion
        u_eeg = torch.rand(z_eeg.size(0), device=device)  # placeholder
        sigma_eeg = torch.rand(z_eeg.size(0), device=device)
        u_text = torch.rand(z_text.size(0), device=device)
        sigma_text = torch.rand(z_text.size(0), device=device)
        z_shared = hyperfusion(z_eeg, z_text, u_eeg, sigma_eeg, u_text, sigma_text)
        bin_logits, ord_logits = classifier(z_shared)
        # Loss
        total_loss, _, _ = combined_loss(bin_logits, ord_logits, bin_target, ord_target, contrast_loss + adv_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return total_loss.item()
