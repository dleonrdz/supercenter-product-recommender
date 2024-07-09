from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supercenter_product_recommender.vector_storage import load_faiss_index

def faiss_retrieval(db_path, transformer_name, k=5):
    db = load_faiss_index(path=db_path, transformer=transformer_name)
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever