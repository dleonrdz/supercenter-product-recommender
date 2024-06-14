from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def calculate_similarity(order_embeddings, product_embeddings):
    # Calculate the cosine similarity between each order embedding and all product embeddings
    similarities = cosine_similarity(order_embeddings, product_embeddings)
    return similarities

def get_top_n_recommendations(similarity_matrix, product_ids, n=5):
    recommendations = []
    for idx, similarities in enumerate(similarity_matrix):
        top_n_indices = np.argsort(similarities)[-n:][::-1]
        top_n_product_ids = [product_ids[i] for i in top_n_indices]
        recommendations.append((idx, top_n_product_ids))
    return recommendations
