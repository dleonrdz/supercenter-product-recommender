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
prod_cols = ['product_id', 'product_name', 'department']
df_orders, df_products = data_preparation(orders_df,
                                          ord_cols,
                                          products_df,
                                          prod_cols)

print('Creating vectorstores...')
create_faiss_index(df=df_products,
                   id_column='product_id',
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='products_index_pt')

create_faiss_index(df=df_orders,
                   id_column='order_id',
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='orders_index_pt')

print('Embedding Pipeline Complete.')

