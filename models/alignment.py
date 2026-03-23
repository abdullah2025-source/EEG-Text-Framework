import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class CLEP(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z_eeg, z_text):
        z_eeg = F.normalize(z_eeg, dim=1)
        z_text = F.normalize(z_text, dim=1)
        sim = torch.matmul(z_eeg, z_text.T) / self.temp
        labels = torch.arange(len(z_eeg)).to(z_eeg.device)
        loss = F.cross_entropy(sim, labels)
        return loss

class DomainAdversarial(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, z, modality_label):
        z_adv = GradientReversalLayer.apply(z)
        logits = self.disc(z_adv)
        loss = F.cross_entropy(logits, modality_label)
        return loss
