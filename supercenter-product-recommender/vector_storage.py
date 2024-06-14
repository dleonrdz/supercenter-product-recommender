import numpy as np
from langchain.vectorstores import FAISS
import os
# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_faiss_index(model, embeddings,ids, name):
    path = os.path.join(PROJECT_ROOT, f'vectorstores/{name}')
    vector_store = FAISS(embedding_function=model)
    vector_store.add(embeddings,ids)
    vector_store.save_local()