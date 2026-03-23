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
        x = self.dilated_conv(x)          # (batch, d_model, time)
        x = x.permute(0, 2, 1)            # (batch, time, d_model)
        return self.transformer(x).mean(dim=1)   # (batch, d_model)

class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # h: (n_nodes, in_features), adj: (n_nodes, n_nodes) adjacency matrix (mask)
        Wh = self.W(h)                     # (n_nodes, out_features)
        Wh_i = Wh.unsqueeze(1).expand(-1, Wh.size(0), -1)   # (n_nodes, n_nodes, out)
        Wh_j = Wh.unsqueeze(0).expand(Wh.size(0), -1, -1)   # (n_nodes, n_nodes, out)
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        attention = F.softmax(e * adj, dim=1)
        return torch.matmul(attention, Wh)   # (n_nodes, out_features)

class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        eeg_channels = config['eeg_channels']
        self.tct = TCT(eeg_channels, d_model, nhead, num_layers)
        self.gat = GATv2Layer(eeg_channels, d_model)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_proj = nn.Linear(768, d_model)
        self.moe_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, eeg_signal, eeg_spectrogram, adjacency):
        # eeg_signal: (batch, channels, time)
        # eeg_spectrogram: (batch, channels, freq, time) — we need to feed ViT
        # For ViT, we'll reshape spectrogram to image-like (batch, 3, H, W) but here we simplify:
        # We'll use the ViT model expecting 3-channel images; we can convert spectrogram to 3-channel by repeating.
        # In practice, you'd adapt ViT for EEG spectrograms. For simplicity, we average over channels.
        # (We'll keep the logic as in the paper: spectrogram patches are processed by ViT.)
        # For demonstration, we treat the spectrogram as a 2D image per channel and average across channels.
        # You can replace this with a proper implementation.
        batch_size = eeg_signal.size(0)
        # Average spectrogram over channels to get (batch, freq, time)
        spec_avg = eeg_spectrogram.mean(dim=1)  # (batch, freq, time)
        # Resize to 224x224 (ViT expects 224x224)
        spec_resized = F.interpolate(spec_avg.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).repeat(1,3,1,1)
        vit_out = self.vit(pixel_values=spec_resized).last_hidden_state[:,0]  # (batch, 768)
        vit_out = self.vit_proj(vit_out)  # (batch, d_model)

        tct_out = self.tct(eeg_signal)  # (batch, d_model)

        # GAT expects node features of shape (nodes, features). Here we treat each channel as a node.
        # We'll pool across time to get node features: (batch, channels) then average over batch for adjacency?
        # Actually GAT processes a single graph per sample. We'll loop over batch.
        gat_out_list = []
        for i in range(batch_size):
            node_feats = eeg_signal[i].mean(dim=1)  # (channels,)
            adj = adjacency[i]  # (channels, channels)
            gat_out_i = self.gat(node_feats, adj)  # (channels, d_model)
            gat_out_i = gat_out_i.mean(dim=0)  # (d_model,)
            gat_out_list.append(gat_out_i)
        gat_out = torch.stack(gat_out_list)  # (batch, d_model)

        alphas = F.softmax(self.moe_weights, dim=0)
        return alphas[0]*tct_out + alphas[1]*gat_out + alphas[2]*vit_out
