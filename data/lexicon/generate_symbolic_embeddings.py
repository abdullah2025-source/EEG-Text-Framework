"""
Generate symbolic embeddings for suicide-related terms using TransE.
Requires: pykg2vec, pandas, nltk (or similar).
"""
import os
import numpy as np
import pandas as pd
from pykg2vec.common import Importer
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.batch_data import TestTriple

# ----------------------------------------------------------------------
# Build knowledge graph from DSM-5 and suicide lexicon
# ----------------------------------------------------------------------

def build_kg(lexicon_file, dsm_file):
    """
    Create a set of triples (head, relation, tail) from the given lexicons.
    For example: (term, 'is_symptom_of', 'disorder')
    This is a simplified placeholder; in practice you would use a real KG.
    """
    # Load lexicon
    with open(lexicon_file, 'r') as f:
        terms = [line.strip() for line in f.readlines()]
    # Load DSM-5 disorders
    dsm = pd.read_csv(dsm_file)  # expects at least 'disorder' and 'symptom' columns

    triples = []
    # Add triples like (term, 'related_to', 'suicide')
    for term in terms:
        triples.append((term, 'related_to', 'suicide'))
    # Add DSM symptom-disorder relations
    for _, row in dsm.iterrows():
        triples.append((row['symptom'], 'is_symptom_of', row['disorder']))
    return triples

def train_transE(triples, embed_dim=64):
    """
    Train TransE model using pykg2vec.
    Returns entity embeddings.
    """
    from pykg2vec.data.kgcontroller import KnowledgeGraph
    from pykg2vec.models.model import TransE
    from pykg2vec.utils.trainer import Trainer
    from pykg2vec.common import Importer

    # Prepare data
    kg = KnowledgeGraph()
    kg.load_entity_from_triples(triples)
    kg.load_relation_from_triples(triples)

    # Model
    model = TransE(embed_dim=embed_dim)
    config = {
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.001,
        'device': 'cpu'
    }
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()
    # Get embeddings
    entity_embeddings = model.ent_embeddings.weight.detach().numpy()
    entity_to_idx = kg.ent2idx
    return entity_embeddings, entity_to_idx

def save_embeddings(entity_embeddings, entity_to_idx, output_dir):
    """Save embeddings and mapping."""
    np.save(os.path.join(output_dir, 'symbolic_embeddings.npy'), entity_embeddings)
    with open(os.path.join(output_dir, 'entity_to_idx.json'), 'w') as f:
        import json
        json.dump(entity_to_idx, f)

if __name__ == '__main__':
    # Paths (adjust as needed)
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    lexicon_file = os.path.join(data_dir, 'lexicon', 'suicide_lexicon.txt')
    dsm_file = os.path.join(data_dir, 'lexicon', 'dsm5_symptoms.csv')
    output_dir = os.path.join(data_dir, 'lexicon')
    os.makedirs(output_dir, exist_ok=True)

    # Build knowledge graph
    triples = build_kg(lexicon_file, dsm_file)
    print(f"Built {len(triples)} triples.")

    # Train TransE
    embeddings, mapping = train_transE(triples, embed_dim=64)
    print(f"Trained embeddings shape: {embeddings.shape}")

    # Save
    save_embeddings(embeddings, mapping, output_dir)
    print("Saved symbolic embeddings.")
