from vector_storage import json_to_pinecone_format, upsert_embeddings_to_pincone_index
print('Preparing embeddings in json file...')
pinecone_data = json_to_pinecone_format('product_embeddings_ft/product_embeddings.json')

print('Upserting to pinecone...')
upsert_embeddings_to_pincone_index('supercenter-recommender-system',
                                   pinecone_data,
                                   batch_size=500)
print('Embeddings upserted successfully.')
