#import tensorflow as tf
#import numpy as np
from db_utilities import read_table
from sentence_transformers import SentenceTransformer

def data_preparation(orders, orders_features, products, products_features):

  orders['text_feature'] = orders[orders_features]\
    .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  df_orders = orders.groupby('order_id',
                                as_index=False)['text_feature'].agg(lambda x: ' '.join(x))

  products['text_feature'] = products[products_features]\
    .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

  return df_orders[['order_id','text_feature']], products[['product_id','text_feature']]

def embedding_process(df_orders,df_products, transformer):
  model = SentenceTransformer(transformer)

  order_embeddings = model.encode(df_orders['text_feature'].tolist())
  product_embeddings = model.encode(df_products['text_feature'].tolist())

  #df_orders['embeddings'] = order_embeddings.tolist()
  #df_products['embeddings'] = product_embeddings.tolist()

  return order_embeddings, product_embeddings, model



