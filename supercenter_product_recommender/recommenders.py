import tensorflow as tf
from two_towers_finetuning import TwoTowerModel
print('Loading model...')
model = tf.keras.models.load_model('models/two_tower_architecture_trained', custom_objects={'TwoTowerModel': TwoTowerModel})
print('Model loaded')
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supercenter_product_recommender.vector_storage import load_faiss_index
from supercenter_product_recommender.embedding_process import embedding_process, get_refined_embeddings

"""
This script defines the functions for each recommender we have: recommendation retrieval with
pre-trained model embeddings and the recommender with refined embeddings.
"""

# Loading environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


def get_top_n_recommendations_faiss(text_features, n=5):
    """
    This function retrieves recommendations through FAISS index, that is a local
    vectorstore with pre-trained model embeddings. It takes the text features
    and embed them under the hood for further similarity search within
    product index
    """
    # Initializing faiss retriever
    db = load_faiss_index(path='vectorstores/products_index_pt',
                          transformer='hiiamsid/sentence_similarity_spanish_es')
    retriever = db.as_retriever(search_kwargs={"k": n})

    # Empty list that will store recommendations
    recommendations = []

    # For each text retrieve top n recommendations
    for text in text_features:
        docs = retriever.invoke(text)
        recommendations.append([doc.metadata['id'] for doc in docs])

    return recommendations
def get_top_n_recommendations_pinecone_batch(index_name, texts,n=5):
    """
    This function retrieves recommendations through a pinecone index, which is a cloud
    vectorstore with the refined products embeddings. It takes the text features,
    embed them with the pre-trained model and the refines them with the trained
    two-tower architecture for further similarity search within product index
    """

    # Initializing pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Embeddings with pre-trained model
    initial_embeddings = embedding_process(texts, transformer='hiiamsid/sentence_similarity_spanish_es')

    # Refine embeddings
    refined_embeddings = get_refined_embeddings(model, embeddings=initial_embeddings, tower='order')
    refined_embeddings = [emb.numpy() if hasattr(emb, 'numpy') else np.array(emb) for emb in refined_embeddings]

    # Initialize pinecone index
    index = pc.Index(index_name)

    # Empty list that will store recommendations
    recommendations = []

    # For each text retrieve top n recommendations
    for emb in refined_embeddings:
        docs = index.query(
            vector=emb.tolist(),
            top_k=n,
            include_metadata=True
        )
        recommendations.append([doc['id'] for doc in docs['matches']])

    return recommendations



