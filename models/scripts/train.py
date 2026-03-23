import torch
import yaml
from torch.utils.data import DataLoader
from models.eeg_encoder import EEGEncoder
from models.text_encoder import TextEncoder
from models.alignment import CLEP, DomainAdversarial
from models.fusion import HypernetworkFusion
from models.classifier import ClassifierHead
from training.losses import CombinedLoss
from data.dataset import MultimodalDataset

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets (implement your data loader)
    train_dataset = MultimodalDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Models
    eeg_encoder = EEGEncoder(config['model']).to(device)
    text_encoder = TextEncoder(config['model']).to(device)
    clep = CLEP(temperature=config['model']['temperature'])
    dann = DomainAdversarial(config['model']['d_model'])
    hyperfusion = HypernetworkFusion(config['model']['d_model'])
    classifier = ClassifierHead(config['model']['d_model']).to(device)
    loss_fn = CombinedLoss()

    # Optimizers with different LR
    optimizer = torch.optim.AdamW([
        {'params': eeg_encoder.parameters(), 'lr': config['training']['lr_eeg']},
        {'params': text_encoder.parameters(), 'lr': config['training']['lr_text']},
        {'params': hyperfusion.parameters(), 'lr': config['training']['lr_align']},
        {'params': classifier.parameters(), 'lr': config['training']['lr_align']},
        {'params': clep.parameters(), 'lr': config['training']['lr_align']},
        {'params': dann.parameters(), 'lr': config['training']['lr_align']}
    ])

    for epoch in range(config['training']['epochs']):
        total_loss = 0.0
        for batch in train_loader:
            # batch = (eeg_signal, eeg_spec, adj, text, bin_label, ord_label, mod_label)
            eeg_signal, eeg_spec, adj, text, bin_t, ord_t, mod_t = [x.to(device) for x in batch]

            z_eeg = eeg_encoder(eeg_signal, eeg_spec, adj)
            z_text = text_encoder(text)

            contrast_loss = clep(z_eeg, z_text)
            adv_loss = dann(z_eeg, mod_t) + dann(z_text, mod_t)

            # Placeholder uncertainty & complexity (replace with real estimates)
            u_eeg = sigma_eeg = u_text = sigma_text = torch.ones(z_eeg.size(0), device=device) * 0.1

            z_shared = hyperfusion(z_eeg, z_text, u_eeg, sigma_eeg, u_text, sigma_text)
            bin_logits, ord_logits = classifier(z_shared)

            loss, _, _ = loss_fn(bin_logits, ord_logits, bin_t, ord_t, contrast_loss + adv_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(eeg_encoder.parameters()) + list(text_encoder.parameters()) +
                list(hyperfusion.parameters()) + list(classifier.parameters()),
                config['training']['gradient_clip']
            )
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

if __name__ == '__main__':
    main()
