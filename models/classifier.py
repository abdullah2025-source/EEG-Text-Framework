import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes_bin=2, num_classes_ord=4):
        super().__init__()
        self.decoder = nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True)
        self.bin_head = nn.Linear(d_model, num_classes_bin)
        self.ord_head = nn.Linear(d_model, num_classes_ord)

    def forward(self, z_shared):
        # z_shared: (batch, d_model)
        z_shared = z_shared.unsqueeze(1)          # (batch, 1, d_model)
        decoded = self.decoder(z_shared, z_shared).squeeze(1)  # (batch, d_model)
        bin_logits = self.bin_head(decoded)
        ord_logits = self.ord_head(decoded)
        return bin_logits, ord_logits
