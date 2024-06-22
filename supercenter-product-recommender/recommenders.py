from retrievers import faiss_retrieval
def get_top_n_recommendations_faiss(text_features, n=5):
    retriever = faiss_retrieval(
        db_path='vectorstores/products_index_pt',
        transformer_name='hiiamsid/sentence_similarity_spanish_es',
        k=n
    )
    docs_list = retriever.invoke(text_features)
    recommendations = [[doc.metadata['id'] for doc in docs] for docs in docs_list]
    return recommendations



