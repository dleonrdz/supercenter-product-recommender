#import tensorflow as tf
#import numpy as np
from db_utilities import read_table
from sentence_transformers import SentenceTransformer

def data_preparation(orders, orders_features, products, products_features):

  df_orders = orders[orders_features]
  df_products = products[products_features]

  df_orders['text_feature'] = df_orders.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  df_products['text_feature'] = df_products.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

  return df_orders[['order_id','text_feature']], df_products[['product_id','text_feature']]

def embedding_process(df_orders,df_products, transformer):
  model = SentenceTransformer(transformer)
  df_orders['embeddings'] = model.encode(df_orders['text_feature'])
  df_products['embeddings'] = model.encode(df_products['text_feature'])

  return df_orders, df_products, model



