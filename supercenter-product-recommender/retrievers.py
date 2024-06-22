from sklearn.metrics.pairwise import cosine_similarity
from vector_storage import load_faiss_index
import numpy as np

def faiss_retrieval(db_path, transformer_name, k=5):
    db = load_faiss_index(path=db_path, transformer=transformer_name)
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever
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

def get_embeddings_by_ids(faiss_index, id_to_index, ids):
    embeddings = []
    for id in ids:
        index_pos = id_to_index[id]
        embeddings.append(faiss_index.reconstruct(index_pos))
    return np.array(embeddings)


if __name__ == "__main__":
    db = load_faiss_index(path='vectorstores/orders_index_pt', transformer='hiiamsid/sentence_similarity_spanish_es')
    retriever = db.as_retriever(search_kwargs={"k": 5})
    input = []
    docs = retriever.invoke()
    print(train_embeddings)