from db_utilities import read_table
from embedding_process import data_preparation, embedding_process
from vector_storage import create_faiss_index
import numpy as np

print('Reading data...')
orders_df = read_table('processed_orders_data')
products_df = read_table('processed_products_data')
orders_df['order_id'] = orders_df['order_id'].astype(str)

print('Processing data...')
ord_cols = ['product_id', 'product_name', 'department']
prod_cols = ['product_name', 'department']
df_orders, df_products = data_preparation(orders_df,
                                          ord_cols,
                                          products_df,
                                          prod_cols)

#print('Creating product embeddings...')
#order_embeddings, product_embeddings, model = embedding_process(df_orders,
 #                                                       df_products,
  #                                                      'hiiamsid/sentence_similarity_spanish_es')

print('Creating vectorstores...')
df_products = df_products.head(100)
df_products = df_orders.head(100)
create_faiss_index(df=df_products,
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='pt_products_index')

create_faiss_index(df=df_orders,
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='pt_products_index')

print('Embedding Pipeline Complete.')

