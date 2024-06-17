import tensorflow as tf
import numpy as np
from two_towers_finetuning import TwoTowerModel, train_step
from model_evaluation import train_orders_df
from vector_storage import load_faiss_index
from retrievers import get_embeddings_by_ids

print('Reading data...')
train_orders_df['order_id'] = train_orders_df['order_id'].astype(str)
train_order_ids = list(train_orders_df['order_id'].unique())

print('Retrieving embeddings...')
db_path = 'vectorstores/products_index_pt'
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'
db = load_faiss_index(db_path, transformer_name)
index = db.index
product_embeddings = index.reconstruct_n(0, index.ntotal)
product_embeddings = np.array(product_embeddings)
id_to_index = {id: idx for idx, id in enumerate(train_order_ids)}
train_order_embeddings = get_embeddings_by_ids(index, id_to_index, train_order_ids)


print('Preparing for training...')
model = TwoTowerModel(product_embedding_dim=product_embeddings.shape[1], order_embedding_dim=train_order_embeddings.shape[1])
product_embeddings = np.vstack(product_embeddings)
train_order_embeddings = np.vstack(train_order_embeddings)

optimizer = tf.keras.optimizers.Adam()

print('Training...')
epochs = 10
for epoch in range(epochs):
    loss = train_step(model, product_embeddings, train_order_embeddings, optimizer)
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

model.save('models/two_towers_trained')
