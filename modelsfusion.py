import torch
import torch.nn as nn

class Hypernetwork(nn.Module):
    def __init__(self, d_model, n_modalities=2):
        super().__init__()
        self.fc = nn.Linear(2, d_model)  # input: uncertainty and complexity
        self.out = nn.Linear(d_model, n_modalities)
    def forward(self, u, sigma):
        x = torch.cat([u, sigma], dim=-1)  # (batch, 2)
        h = torch.relu(self.fc(x))
        beta = torch.softmax(self.out(h), dim=-1)
        return beta

class HypernetworkFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hyper = Hypernetwork(d_model)
    def forward(self, z_eeg, z_text, u_eeg, sigma_eeg, u_text, sigma_text):
        beta = self.hyper(torch.stack([u_eeg, u_text], dim=-1), torch.stack([sigma_eeg, sigma_text], dim=-1))
        return beta[:,0:1] * z_eeg + beta[:,1:2] * z_text
