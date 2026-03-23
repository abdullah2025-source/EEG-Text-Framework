import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SymbolicEmbedding(nn.Module):
    def __init__(self, embed_dim, lexicon_embeddings):
        super().__init__()
        # lexicon_embeddings: (vocab_size, embed_dim) precomputed (TransE/RotatE)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(lexicon_embeddings), freeze=False)

    def forward(self, token_ids):
        return self.embed(token_ids)

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['bert_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
        # Load precomputed symbolic embeddings (placeholders)
        self.symbolic = SymbolicEmbedding(config['d_model'], np.zeros((30522, config['d_model'])))
        self.proj = nn.Linear(self.bert.config.hidden_size + config['d_model'], config['d_model'])

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                max_length=128).to(self.bert.device)
        bert_out = self.bert(**tokens).last_hidden_state[:,0]    # (batch, 768)
        sym_out = self.symbolic(tokens['input_ids']).mean(dim=1)  # (batch, d_model)
        combined = torch.cat([bert_out, sym_out], dim=-1)
        return self.proj(combined)
