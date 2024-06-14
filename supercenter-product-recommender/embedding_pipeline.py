from db_utilities import read_table
from embedding_process import data_preparation, embedding_process
from vector_storage import create_faiss_index
import numpy as np

print('Reading data...')
orders_df = read_table('processed_orders_data')
products_df = read_table('processed_products_data')
orders_df['order_id'] = orders_df['order_id'].astype(str)

print('Processing data...')
ord_cols = ['order_id', 'product_id', 'product_name']
prod_cols = ['product_id', 'product_name']
df_orders, df_products = data_preparation(orders_df,
                                          ord_cols,
                                          products_df,
                                          prod_cols)

print('Creating product embeddings...')
orders_tower, products_tower, model = embedding_process(df_orders,
                                                        df_products,
                                                        'hiiamsid/sentence_similarity_spanish_es')

print('Storing embeddings...')
product_embeddings = np.array(products_tower['embeddings'].tolist())
product_ids = products_tower['product_id'].tolist()
orders_embeddings = np.array(orders_tower['embeddings'].tolist())
orders_ids = orders_tower['order_id'].tolist()
create_faiss_index(model, product_embeddings, product_ids, 'product_pre_trained_index')
create_faiss_index(model, orders_embeddings, orders_ids, 'order_pre_trained_index')

print('Embedding Pipeline Complete.')

