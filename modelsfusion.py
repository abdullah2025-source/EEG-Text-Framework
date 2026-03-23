import torch
import torch.nn as nn

class Hypernetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(2, d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, u, sigma):
        # u, sigma: (batch,) - uncertainty and complexity
        x = torch.stack([u, sigma], dim=1)  # (batch,2)
        h = torch.relu(self.fc(x))
        beta = torch.softmax(self.out(h), dim=1)
        return beta  # (batch,2)

class HypernetworkFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hyper = Hypernetwork(d_model)

    def forward(self, z_eeg, z_text, u_eeg, sigma_eeg, u_text, sigma_text):
        # For simplicity, combine uncertainties (e.g., average)
        u_combined = (u_eeg + u_text) / 2
        sigma_combined = (sigma_eeg + sigma_text) / 2
        beta = self.hyper(u_combined, sigma_combined)  # (batch,2)
        return beta[:,0:1] * z_eeg + beta[:,1:2] * z_text
