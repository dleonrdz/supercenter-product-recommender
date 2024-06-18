import tensorflow as tf
import numpy as np
from two_towers_finetuning import TwoTowerModel, train_step
from model_evaluation import train_orders_df
from vector_storage import load_faiss_index
from embedding_process import data_preparation_orders, embedding_process
#from retrievers import get_embeddings_by_ids

print('Reading data...')
train_orders_df['order_id'] = train_orders_df['order_id'].astype(str)
train_order_ids = list(train_orders_df['order_id'].unique())

print('Preparing training data...')
features = ['product_id', 'product_name', 'department']
train_orders_df_prepared = data_preparation_orders(train_orders_df, features)

print('Retrieving product embeddings...')
db_path = 'vectorstores/products_index_pt'
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'
db = load_faiss_index(db_path, transformer_name)
index = db.index
product_embeddings = index.reconstruct_n(0, index.ntotal)
product_embeddings = np.array(product_embeddings)

print('Creating training orders embeddings...')
train_order_embeddings = embedding_process(train_orders_df_prepared, transformer_name)
train_order_embeddings = np.array(train_order_embeddings)

print('Preparing for training...')
embedding_dim = 768  # Assuming the desired output embedding dimension is the same as input dimension
model = TwoTowerModel(embedding_dim=embedding_dim)
optimizer = tf.keras.optimizers.Adam()

print('Training...')
epochs = 10
batch_size = 32

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

model.save('models/two_towers_trained')

