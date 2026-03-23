import torch
import torch.nn as nn
import torch.nn.functional as F

def ordinal_loss(logits, targets, num_classes=4):
    # Simple ordinal regression using cumulative link
    targets = targets.view(-1, 1)
    logits = logits.view(-1, num_classes)
    prob = torch.sigmoid(logits)
    # Convert to cumulative probabilities
    cum_prob = torch.cumsum(prob, dim=1)
    # Binary cross-entropy per threshold
    loss = 0.0
    for k in range(num_classes-1):
        y_k = (targets > k).float()
        loss += F.binary_cross_entropy_with_logits(logits[:,k], y_k)
    return loss / (num_classes-1)

class CombinedLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.bce = nn.CrossEntropyLoss()
    def forward(self, bin_logits, ord_logits, bin_targets, ord_targets, contrast_loss):
        loss_bin = self.bce(bin_logits, bin_targets)
        loss_ord = ordinal_loss(ord_logits, ord_targets)
        total = self.lambda1*loss_bin + self.lambda2*loss_ord + self.lambda3*contrast_loss
        return total, loss_bin, loss_ord
