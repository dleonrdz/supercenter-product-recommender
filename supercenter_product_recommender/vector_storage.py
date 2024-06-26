from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from dotenv import load_dotenv
#from datetime import datetime
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_faiss_index(df, id_column, transformer, name):
    model=HuggingFaceEmbeddings(model_name=transformer)
    path = os.path.join(PROJECT_ROOT, f'vectorstores/{name}')
    metadata = df[[id_column, 'text_feature']].rename(columns={id_column: 'id', 'text_feature': 'text'})\
        .to_dict(orient='records')
    vector_store = FAISS.from_texts(df['text_feature'].tolist(),
                                    model,
                                    ids = df[id_column].tolist(),
                                    metadatas=metadata)
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
def json_to_pinecone_format(json_file):
    json_file_path = os.path.join(PROJECT_ROOT, f'vectorstores/{json_file}')
    print('Reading json file...')
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print('Refactoring to suitable format...')
    pinecone_data = [(item['id'], item['embedding'], {"id": item['id']}) for item in data]
    return pinecone_data
def upsert_embeddings_to_pincone_index(index_name, pinecone_data, batch_size=100):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    for i in range(0, len(pinecone_data), batch_size):
        batch = pinecone_data[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f'Batch {i} upserted')
    print(index.describe_index_stats())








