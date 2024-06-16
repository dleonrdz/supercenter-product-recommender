from sentence_transformers import SentenceTransformer
import os

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_preparation(orders, orders_features, products, products_features):

  orders['text_feature'] = orders[orders_features]\
    .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
  df_orders = orders.groupby('order_id',
                                as_index=False)['text_feature'].agg(lambda x: ' '.join(x))

  products['text_feature'] = products[products_features]\
    .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

  return df_orders[['order_id','text_feature']], products[['product_id','text_feature']]

def embedding_process(df_orders,df_products, transformer):
  path = os.path.join(PROJECT_ROOT, 'data/')
  model = SentenceTransformer(transformer)

  order_embeddings = model.encode(df_orders['text_feature'].tolist())
  product_embeddings = model.encode(df_products['text_feature'].tolist())

  df_orders['embedding'] = order_embeddings.tolist()
  df_products['embedding'] = product_embeddings.tolist()

  df_orders.to_json(f'{path}orders_embeddings_pt.json', orient='records', lines=True)
  df_products.to_json(f'{path}products_embeddings_pt.json', orient='records', lines=True)

  return order_embeddings, product_embeddings, model



