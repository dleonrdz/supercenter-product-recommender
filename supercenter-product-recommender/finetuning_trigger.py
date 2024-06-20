import tensorflow as tf
import numpy as np
from two_towers_finetuning import TwoTowerModel, train_step
from model_evaluation import train_orders_df
from vector_storage import load_faiss_index
from embedding_process import data_preparation_orders, embedding_process
import pickle
#from retrievers import get_embeddings_by_ids

print('Reading data...')
train_orders_df['order_id'] = train_orders_df['order_id'].astype(str)
train_order_ids = list(train_orders_df['order_id'].unique())

print('Preparing training data...')
features = ['product_id', 'product_name', 'department']
train_orders_df_prepared = data_preparation_orders(train_orders_df, features)

print('Retrieving product embeddings...')
db_path_products = 'vectorstores/products_index_pt'
db_path_orders = 'vectorstores/orders_index_pt'
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'
db_products = load_faiss_index(db_path_products, transformer_name)
db_orders = load_faiss_index(db_path_orders, transformer_name)
index_products = db_products.index
index_orders = db_orders.index
product_embeddings = index_products.reconstruct_n(0, index_products.ntotal)
product_embeddings = np.array(product_embeddings)

print('Retrieving orders embeddings...')
orders_embeddings = index_orders.reconstruct_n(0, index_orders.ntotal)
orders_embeddings = np.array(orders_embeddings)

with open('vectorstores/orders_index_pt/index.pkl', 'rb') as f:
    index_content = pickle.load(f)

index_to_order_id = index_content[1]

order_id_to_index = {v: k for k, v in index_to_order_id.items()}

train_indices = [order_id_to_index[id] for id in train_order_ids if id in order_id_to_index]
train_order_embeddings = np.array(orders_embeddings[train_indices])

#print('Creating training orders embeddings...')
#train_order_embeddings = embedding_process(train_orders_df_prepared, transformer_name)
#train_order_embeddings = np.array(train_order_embeddings)

print('Preparing for training...')
product_embedding_dim = 768
order_embedding_dim = 768
model = TwoTowerModel(product_embedding_dim, order_embedding_dim)
optimizer = tf.keras.optimizers.Adam()

print('Training...')
epochs = 10
batch_size = 32

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, len(product_embeddings), batch_size):
        product_batch = product_embeddings[i:i + batch_size]
        order_batch = train_order_embeddings[i:i + batch_size]

        if product_batch.shape[0] != order_batch.shape[0]:
            min_batch_size = min(product_batch.shape[0], order_batch.shape[0])
            product_batch = product_batch[:min_batch_size]
            order_batch = order_batch[:min_batch_size]

        loss = train_step(model, product_batch, order_batch, optimizer)
        print(f"Batch {i // batch_size + 1}/{len(product_embeddings) // batch_size}: Loss = {loss.numpy()}")

print('Saving model...')
model.save('models/two_towers_trained.keras')

'''
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, min(len(product_embeddings), len(train_order_embeddings)), batch_size):
        product_batch = product_embeddings[i:i + batch_size]
        order_batch = train_order_embeddings[i:i + batch_size]
        if product_batch.shape[0] != order_batch.shape[0]:
            min_batch_size = min(product_batch.shape[0], order_batch.shape[0])
            product_batch = product_batch[:min_batch_size]
            order_batch = order_batch[:min_batch_size]

        loss = train_step(model, product_batch, order_batch, optimizer)
        print(f"Batch {i // batch_size + 1}/{len(product_embeddings) // batch_size}: Loss = {loss.numpy()}")

'''