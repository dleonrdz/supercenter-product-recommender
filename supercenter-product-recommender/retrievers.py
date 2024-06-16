from sklearn.metrics.pairwise import cosine_similarity
from vector_storage import load_faiss_index
import numpy as np
from db_utilities import read_table
from embedding_process import data_preparation

def faiss_retrieval(db_path, transformer_name, k=5):
    db = load_faiss_index(path=db_path, transformer=transformer_name)
    return db.as_retriever(search_kwargs={"k": k})

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


if __name__ == "__main__":
    orders_df = read_table('processed_orders_data')
    products_df = read_table('processed_products_data')
    ord_cols = ['product_id', 'product_name', 'department']
    prod_cols = ['product_name', 'department']
    df_orders, df_products = data_preparation(orders_df,
                                              ord_cols,
                                              products_df,
                                              prod_cols)
    df_orders = df_orders.head(1)
    query = df_orders['text_feature'].iloc[0]
    retriever = faiss_retrieval('vectorstores/products_index_pt', 'hiiamsid/sentence_similarity_spanish_es')
    response = retriever.invoke(query)
    print(response)