from sentence_transformers import SentenceTransformer
import tensorflow as tf
import os
import numpy as np
import pandas as pd

"""
This script defines through functions the pre-processing steps needed in order
to create and save.

"""

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_preparation_orders(df_orders):
    """
        This function takes the input of the PoC data and pre-process it in a suitable
        format for embeddings creation and further recommendations retrieval
    """
    df_orders['text_feature'] = 'Producto ' + df_orders['cart_inclusion_order'].astype(str) + ': con id ' + \
    df_orders['product_id'].astype(str) + ' y nombre ' + df_orders['product_name'].astype(str) + \
    ' del departmento ' + df_orders['department'].astype(str) + \
    ' y reordenado ' + np.where(df_orders['reordered']==1, 'sí', 'no')

    df_orders = df_orders.groupby('order_id',
                                  as_index=False)['text_feature'].agg(lambda x: ','.join(x))

    return df_orders[['order_id', 'text_feature']]

def data_preparation_orders_test(df_orders):
    """
    This function is specifically designed to pre-process the orders data for the two-tower architecture
    training process. In order to match each query (or cart) with each of the items as the co-ocurrence
    within each cart, the string feature must match the information the cart actually had until the moment
    of adding that product.
    """
    # Create the text feature for each feature
    df_orders['product_feature'] = 'Producto ' + df_orders['cart_inclusion_order'].astype(str) + ': con id ' + \
                                   df_orders['product_id'].astype(str) + ' y nombre ' + df_orders['product_name'].astype(str) + \
                                   ' del departmento ' + df_orders['department'].astype(str) + \
                                   ' y reordenado ' + np.where(df_orders['reordered'] == 1, 'sí', 'no')

    # Concatenate the products within the order until the corresponding product
    df_orders['order_text_feature_val'] = df_orders.groupby('order_id')['product_feature'] \
        .apply(lambda x: (x + ' ').cumsum().str.strip()).reset_index(level=0, drop=True)

    # Shift one row to simulate the cart a moment previous to add the corresponding row product
    df_orders['order_text_feature'] = df_orders.groupby('order_id')['order_text_feature_val']\
        .shift().\
        fillna('Primer producto')

    # Create the product catalog version string feature of the product
    df_orders['product_text_feature'] = 'Producto: con id ' + \
                                        df_orders['product_id'].astype(str) + ' ' + df_orders['product_name'].astype(str) + \
                                        ' del departmento ' + df_orders['department'].astype(str)

    return df_orders[['id_interaction', 'order_id', 'order_text_feature', 'order_text_feature_val','cart_inclusion_order','product_text_feature']]

def data_preparation_products(df_products):
    """
    This function is specifically designed to pre-process the products data for further embedding process.
    As catalog is an static dataset, no differentiation is needed between training and other processes for this
    pre-processing steps
    """

    # Create the product catalog version string feature
    df_products['text_feature'] = 'Producto: con id ' + \
                                  df_products['product_id'].astype(str) + ' ' + df_products['product_name'].astype(str) + \
                                  ' del departmento ' + df_products['department'].astype(str)
    return df_products[['product_id','text_feature']]

def embedding_process(texts, transformer):
    """
    This function computes and returns the embeddings of a given list of test with a pre-trained
    model
    """
    path = os.path.join(PROJECT_ROOT, 'data/')
    model = SentenceTransformer(transformer)
    embeddings = model.encode(texts)
    return embeddings

def get_refined_embeddings(model, embeddings, tower = 'product'):
    """
    This function takes embeddings and refined either the orders or products-related
    ones through an already trained model and return these refined embeddings
    """
    dummy_embeddings = tf.zeros_like(embeddings)
    tf_embeddings = tf.constant(embeddings)
    if tower == 'product':
        inputs = {'order_embedding': dummy_embeddings, 'product_embedding': tf_embeddings}
        _, refined_embeddings = model(inputs)
    elif tower == 'order':
        inputs = {'order_embedding': tf_embeddings, 'product_embedding': dummy_embeddings}
        refined_embeddings, _ = model(inputs)

    return refined_embeddings
