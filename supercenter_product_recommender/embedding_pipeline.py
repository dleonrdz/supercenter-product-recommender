from db_utilities import read_table, write_table
from embedding_process import data_preparation_orders, data_preparation_products, embedding_process, data_preparation_orders_test
from vector_storage import create_faiss_index
import numpy as np
"""
This script is only for call and run the needed steps in order to create the initial 
embeddings with the pre-trained fine tuned version of Distl BERT on Spanish language.

The functions are defined on its respective scripts
"""
print('Reading data...')
orders_df = read_table('processed_orders_data')
products_df = read_table('processed_products_data')
orders_df['order_id'] = orders_df['order_id'].astype(str)

print('Splitting train and test_orders...')
# 70% is suitable because of the dimensions we have
train_size = 0.7

# Splis is applied at an order level
train_order_ids = orders_df['order_id'].drop_duplicates().sample(frac=train_size, random_state=42)
train_df = orders_df[orders_df['order_id'].isin(train_order_ids)]
test_df = orders_df[(~orders_df['order_id'].isin(train_order_ids))]

# Saving the split datasets for reproducibility
write_table(train_df, 'training_orders')
write_table(test_df, 'test_orders')
print(f'Training shape = {train_df.shape[0]}')


print('Processing data...')
# Pre-process steps for creating embeddings are applied to each dataset
train_df_prepared = data_preparation_orders_test(train_df)
df_products = data_preparation_products(products_df)

print(f'Creating vectorstores...')
# A FAISS index is created for orders embeddings and another one for products embeddings

create_faiss_index(df=train_df_prepared,
                   text_column='order_text_feature',
                   id_column='id_interaction',
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='training_orders_index_pt')

create_faiss_index(df=df_products,
                   text_column='text_feature',
                   id_column='order_id',
                   transformer='hiiamsid/sentence_similarity_spanish_es',
                   name='products_index_pt')

print('Embedding Pipeline Complete.')

