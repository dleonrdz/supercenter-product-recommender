from retrievers import faiss_retrieval
from pinecone import Pinecone
import os
from embedding_process import embedding_process, get_refined_embeddings
from dotenv import load_dotenv
from two_towers_finetuning import TwoTowerModel
import tensorflow as tf
import numpy as np

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
with tf.keras.utils.custom_object_scope({'TwoTowerModel': TwoTowerModel}):
    model = tf.keras.models.load_model('models/two_towers_trained.keras')

def get_top_n_recommendations_faiss(text_features, n=5):
    retriever = faiss_retrieval(
        db_path='vectorstores/products_index_pt',
        transformer_name='hiiamsid/sentence_similarity_spanish_es',
        k=n
    )
    recommendations = []
    for text in text_features:
        docs = retriever.invoke(text)
        recommendations.append([doc.metadata['id'] for doc in docs])

    return recommendations
def get_top_n_recommendations_pinecone_batch(index_name, texts, n=5):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    initial_embeddings = embedding_process(texts, transformer='hiiamsid/sentence_similarity_spanish_es')

    with tf.keras.utils.custom_object_scope({'TwoTowerModel': TwoTowerModel}):
        model = tf.keras.models.load_model('models/two_towers_trained.keras')

    refined_embeddings = get_refined_embeddings(model, embeddings=initial_embeddings, tower='order')
    refined_embeddings = [emb.numpy() if hasattr(emb, 'numpy') else np.array(emb) for emb in refined_embeddings]

    index = pc.Index(index_name)
    recommendations = []

    for emb in refined_embeddings:
        docs = index.query(
            vector=emb.tolist(),
            top_k=n,
            include_metadata=True
        )
        recommendations.append([doc['id'] for doc in docs['matches']])

    return recommendations



