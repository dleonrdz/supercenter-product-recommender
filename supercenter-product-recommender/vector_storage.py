from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_faiss_index(df, id_column, transformer, name):
    #model = SentenceTransformer(transformer)
    model=HuggingFaceEmbeddings(model_name=transformer)
    path = os.path.join(PROJECT_ROOT, f'vectorstores/{name}')
    vector_store = FAISS.from_texts(df['text_feature'].tolist(), model, ids = df[id_column].tolist())
    vector_store.save_local(path)

def load_faiss_index(path, transformer):
    model = HuggingFaceEmbeddings(model_name=transformer)
    return FAISS.load_local(path, model, allow_dangerous_deserialization= True)


def custom_embeddings_to_json(embeddings, ids, tower):
    if len(embeddings) != len(ids):
        raise ValueError("Embeddings and IDs must have the same length")
    embeddings_list = embeddings.numpy().tolist() if isinstance(embeddings,
                                                                tf.Tensor) else embeddings.tolist() if isinstance(
        embeddings, np.ndarray) else embeddings
    embeddings_list = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                       embeddings_list]

    df = pd.DataFrame({'id': ids, 'embedding': embeddings_list})
    result = df.to_dict(orient='records')

    json_dir = f'vectorstores/{tower}_embeddings_ft/'
    os.makedirs(json_dir, exist_ok=True)

    json_path = os.path.join(json_dir, f'{tower}_embeddings.json')
    with open(json_path, 'w') as f:
        json.dump(result, f)

    print(f"Embeddings and IDs have been stored in {json_path}")
