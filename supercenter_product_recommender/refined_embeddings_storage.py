import tensorflow as tf
from two_towers_finetuning import TwoTowerModel
print('Loading model...')
model = tf.keras.models.load_model('models/two_tower_architecture', custom_objects={'TwoTowerModel': TwoTowerModel})
print('Model loaded')
from vector_storage import custom_embeddings_to_json, json_to_pinecone_format, upsert_embeddings_to_pincone_index
from embedding_process import get_refined_embeddings
from vector_storage import load_faiss_index
import numpy as np


"""
This script is only for call and run the needed steps in order to refine the products (candidates)
embeddings and store them first as json file (as back-up) and then in pineocone vectorstore.

The functions are defined on its respective scripts
"""

print('Retrieving pre-trained product embeddings...')
# Initializing FAISS index
db_path_products = 'vectorstores/products_index_pt'
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'
db_products = load_faiss_index(db_path_products, transformer_name)
index_products = db_products.index
product_embeddings = index_products.reconstruct_n(0, index_products.ntotal)
product_embeddings = np.array(product_embeddings)
product_ids = list(db_products.docstore._dict.keys())

print('Fine-tuning embeddings...')
#Refining embeddings
refined_embeddings = get_refined_embeddings(model,
                                            embeddings=product_embeddings)

print('Storing refined embeddings...')
# Storing as json
custom_embeddings_to_json(embeddings=refined_embeddings,ids=product_ids,tower='product')

print('Refined embeddings stored successfully as json.')

# Parsing to pinecone format
print('Preparing embeddings for vectordb uploading...')
pinecone_data = json_to_pinecone_format('product_embeddings_ft/product_embeddings.json')

# Saving to pinecone
print('Upserting to pinecone...')
upsert_embeddings_to_pincone_index('supercenter-recommender-system',
                                   pinecone_data,
                                   batch_size=500)
print('Embeddings upserted successfully.')

