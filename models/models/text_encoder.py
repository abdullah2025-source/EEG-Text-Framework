import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
import os

class SymbolicEmbedding(nn.Module):
    def __init__(self, embed_dim, lexicon_dir):
        super().__init__()
        # Load precomputed symbolic embeddings
        emb_file = os.path.join(lexicon_dir, 'symbolic_embeddings.npy')
        mapping_file = os.path.join(lexicon_dir, 'entity_to_idx.json')
        if os.path.exists(emb_file) and os.path.exists(mapping_file):
            embeddings = np.load(emb_file)
            with open(mapping_file, 'r') as f:
                self.entity_to_idx = json.load(f)
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float32), freeze=False)
            self.vocab_size = embeddings.shape[0]
        else:
            # Fallback: random embeddings
            self.vocab_size = 1000  # placeholder
            self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        # token_ids: (batch, seq_len) - but we need to map to entity indices.
        # In practice, we need a mapping from tokens to entities. For simplicity, we'll just use token IDs directly.
        # This is a placeholder; a real implementation would map tokens to the entity vocabulary.
        return self.embedding(token_ids)

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        bert_model = config['bert_model']
        self.bert = AutoModel.from_pretrained(bert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.symbolic = SymbolicEmbedding(d_model, config.get('lexicon_dir', 'data/lexicon'))
        self.proj = nn.Linear(self.bert.config.hidden_size + d_model, d_model)

    def forward(self, texts):
        # texts: list of strings
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(self.bert.device) for k, v in tokens.items()}
        bert_out = self.bert(**tokens).last_hidden_state[:,0]  # (batch, 768)
        # Symbolic embeddings: we need to map each token to an entity embedding and average.
        # Here we'll just take the average of token embeddings from the symbolic module.
        token_ids = tokens['input_ids']  # (batch, seq_len)
        sym_out = self.symbolic(token_ids).mean(dim=1)  # (batch, d_model)
        combined = torch.cat([bert_out, sym_out], dim=-1)  # (batch, 768+d_model)
        return self.proj(combined)  # (batch, d_model)
