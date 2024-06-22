from sentence_transformers import SentenceTransformer
import tensorflow as tf
import os

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_preparation_orders(df_orders):

  df_orders['text_feature'] = 'Product ' + df_orders['cart_inclusion_order'].astype(str) + ': ' + \
    df_orders['product_id'].astype(str) + ' ' + df_orders['product_name'].astype(str) + \
    ' from department ' + df_orders['department'].astype(str) + \
    ' and reordered ' + df_orders['reordered'].astype(str)

  df_orders = df_orders.groupby('order_id',
                                as_index=False)['text_feature'].agg(lambda x: ','.join(x))

  return df_orders[['order_id', 'text_feature']]

def data_preparation_products(df_products):
  df_products['text_feature'] = 'Product: ' + \
                              df_products['product_id'].astype(str) + ' ' + df_products['product_name'].astype(str) + \
                              ' from department ' + df_products['department'].astype(str)

  return df_products[['product_id','text_feature']]

def embedding_process(df, transformer):
  path = os.path.join(PROJECT_ROOT, 'data/')
  model = SentenceTransformer(transformer)
  embeddings = model.encode(df['text_feature'].tolist())
  return embeddings

def get_refined_embeddings(model, embeddings, tower = 'product'):
  if tower == 'product':
    inputs = {'product_input': embeddings, 'order_input': tf.zeros_like(embeddings)}
    refined_embeddings, _ = model(inputs)
  elif tower == 'order':
    inputs = {'product_input': tf.zeros_like(embeddings), 'order_input': embeddings}
    _, refined_embeddings = model(inputs)
  return refined_embeddings



