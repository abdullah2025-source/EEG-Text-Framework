import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SymbolicEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, lexicon_file):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Load lexicon and initialize TransE/RotatE embeddings (simplified)
        # In practice, you would load precomputed embeddings
    def forward(self, token_ids):
        return self.embed(token_ids)

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self.symbolic = SymbolicEmbedding(len(self.tokenizer), config.d_model, config.lexicon_path)
        self.proj = nn.Linear(self.bert.config.hidden_size + config.d_model, config.d_model)
    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        bert_out = self.bert(**tokens).last_hidden_state[:,0]  # (batch, 768)
        sym_out = self.symbolic(tokens['input_ids']).mean(dim=1)  # (batch, d_model)
        combined = torch.cat([bert_out, sym_out], dim=-1)
        return self.proj(combined)
