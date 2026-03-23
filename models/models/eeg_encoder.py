import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=dilation*(kernel_size-1)//2, dilation=dilation)
    def forward(self, x):
        return self.conv(x)

class TCT(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.dilated_conv = DilatedConv1D(input_dim, d_model, kernel_size=3, dilation=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.dilated_conv(x)                     # (batch, d_model, time)
        x = x.permute(0, 2, 1)                       # (batch, time, d_model)
        return self.transformer(x).mean(dim=1)       # (batch, d_model)

class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)                               # (N, out)
        Wh_i = Wh.unsqueeze(1).expand(-1, Wh.size(0), -1)   # (N, N, out)
        Wh_j = Wh.unsqueeze(0).expand(Wh.size(0), -1, -1)   # (N, N, out)
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        attention = F.softmax(e * adj, dim=1)
        return torch.matmul(attention, Wh)

class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tct = TCT(config['eeg_channels'], config['d_model'],
                       config['nhead'], config['num_layers'])
        self.gat = GATv2Layer(config['eeg_channels'], config['d_model'])
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_proj = nn.Linear(768, config['d_model'])
        self.moe_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, eeg_signal, eeg_spectrogram, adj):
        # eeg_signal: (batch, channels, time)
        tct_out = self.tct(eeg_signal)               # (batch, d_model)

        gat_in = eeg_signal.permute(0,2,1).mean(dim=1)  # (batch, channels)
        gat_out = self.gat(gat_in, adj)               # (batch, d_model)

        vit_out = self.vit(eeg_spectrogram).last_hidden_state[:,0]  # (batch, 768)
        vit_out = self.vit_proj(vit_out)              # (batch, d_model)

        alphas = F.softmax(self.moe_weights, dim=0)
        return alphas[0]*tct_out + alphas[1]*gat_out + alphas[2]*vit_out
