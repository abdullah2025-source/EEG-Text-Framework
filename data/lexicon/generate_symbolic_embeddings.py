"""
Generate symbolic embeddings for suicide-related terms using TransE.
This script uses the pykg2vec library. If not available, it will output random embeddings as a placeholder.
"""
import os
import numpy as np
import pandas as pd
import json

def build_kg(lexicon_file, dsm_file):
    """Build triples from lexicon and DSM-5 symptom-disorder relations."""
    # Load lexicon
    with open(lexicon_file, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f.readlines() if line.strip()]
    # Load DSM-5 data
    dsm = pd.read_csv(dsm_file)
    triples = []
    # Lexicon triples: (term, 'related_to', 'suicide')
    for term in terms:
        triples.append((term, 'related_to', 'suicide'))
    # DSM triples: (symptom, 'is_symptom_of', disorder)
    for _, row in dsm.iterrows():
        triples.append((row['symptom'], 'is_symptom_of', row['disorder']))
    return triples

def train_transE(triples, embed_dim=64, epochs=100):
    """Train TransE and return entity embeddings."""
    try:
        from pykg2vec.data.kgcontroller import KnowledgeGraph
        from pykg2vec.models.model import TransE
        from pykg2vec.utils.trainer import Trainer

        # Build KG
        kg = KnowledgeGraph()
        kg.load_entity_from_triples(triples)
        kg.load_relation_from_triples(triples)

        model = TransE(embed_dim=embed_dim)
        config = {
            'epochs': epochs,
            'batch_size': 128,
            'learning_rate': 0.001,
            'device': 'cpu'
        }
        trainer = Trainer(model, config)
        trainer.build_model()
        trainer.train_model()
        embeddings = model.ent_embeddings.weight.detach().numpy()
        ent_to_idx = kg.ent2idx
        return embeddings, ent_to_idx
    except ImportError:
        print("pykg2vec not installed. Using random embeddings as placeholder.")
        # Get unique entities
        entities = set()
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
        ent_list = list(entities)
        idx_map = {ent: i for i, ent in enumerate(ent_list)}
        embeddings = np.random.randn(len(ent_list), embed_dim)
        return embeddings, idx_map

def save_embeddings(embeddings, ent_to_idx, output_dir):
    np.save(os.path.join(output_dir, 'symbolic_embeddings.npy'), embeddings)
    with open(os.path.join(output_dir, 'entity_to_idx.json'), 'w') as f:
        json.dump(ent_to_idx, f)

if __name__ == '__main__':
    # Paths relative to this script
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', '..', 'data')
    lexicon_file = os.path.join(data_dir, 'lexicon', 'suicide_lexicon.txt')
    dsm_file = os.path.join(data_dir, 'lexicon', 'dsm5_symptoms.csv')
    output_dir = os.path.join(data_dir, 'lexicon')
    os.makedirs(output_dir, exist_ok=True)

    print("Building knowledge graph...")
    triples = build_kg(lexicon_file, dsm_file)
    print(f"Built {len(triples)} triples.")

    print("Training TransE...")
    embeddings, mapping = train_transE(triples, embed_dim=64, epochs=100)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Saving embeddings...")
    save_embeddings(embeddings, mapping, output_dir)
    print("Done.")
