from vector_storage import load_faiss_index
import numpy as np
import tensorflow as tf
from vector_storage import custom_embeddings_to_json
from embedding_process import get_refined_embeddings
from two_towers_finetuning import TwoTowerModel

print('Retrieving pre-trained product embeddings...')
db_path_products = 'vectorstores/products_index_pt'
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'
db_products = load_faiss_index(db_path_products, transformer_name)
index_products = db_products.index
product_embeddings = index_products.reconstruct_n(0, index_products.ntotal)
product_embeddings = np.array(product_embeddings)
product_ids = list(db_products.docstore._dict.keys())

print('Fine-tuning embeddings...')
with tf.keras.utils.custom_object_scope({'TwoTowerModel': TwoTowerModel}):
    model = tf.keras.models.load_model('models/two_towers_trained.keras')

refined_embeddings = get_refined_embeddings(model,
                                            embeddings=product_embeddings)

print('Storing refined embeddings...')
custom_embeddings_to_json(embeddings=refined_embeddings,ids=product_ids,tower='product')

print('Refined embeddings stored successfully.')


