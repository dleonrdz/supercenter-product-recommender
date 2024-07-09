import tensorflow as tf
import numpy as np
from two_towers_finetuning import TwoTowerModel
from vector_storage import load_faiss_index
import pickle
from db_utilities import read_table
import matplotlib.pyplot as plt

"""
This script is only for call and run the needed steps in order to train the two-tower 
architectures by using the already created embeddings.

The functions are defined on its respective scripts
"""

# Reading training orders
print('Reading data...')
train_orders_df = read_table('training_orders')
train_orders_df['order_id'] = train_orders_df['order_id'].astype(str)

# Initializing vectorstores with stored embeddings
# Paths
db_path_products = 'vectorstores/products_index_pt'
db_path_orders = 'vectorstores/training_orders_index_pt'

# Pre-trained model used
transformer_name = 'hiiamsid/sentence_similarity_spanish_es'

# Loading
db_products = load_faiss_index(db_path_products, transformer_name)
db_orders = load_faiss_index(db_path_orders, transformer_name)
index_products = db_products.index
index_orders = db_orders.index
print('Retrieving product embeddings...')
# Reading all product embeddings
product_embeddings = index_products.reconstruct_n(0, index_products.ntotal)
product_embeddings = np.array(product_embeddings)

# This part will generate a dictionary with the respective product id to embedding
# in order to be able to match the product embedding to all the corresponding orders

with open('vectorstores/products_index_pt/index.pkl', 'rb') as f:
    index_content = pickle.load(f)

index_to_product_id = index_content[1]
product_id_to_embedding = {v: product_embeddings[k] for k, v in index_to_product_id.items()}

print('Retrieving orders embeddings...')
# Reading all traininf orders embeddings and storing them as list
orders_embeddings = index_orders.reconstruct_n(0, index_orders.ntotal)
orders_embeddings_list = orders_embeddings.tolist()

print('Preparing for training...')
# Adding corresponding embeddings to training dataset to match each co-ocurrence order-product
train_orders_df['orders_embeddings'] = orders_embeddings_list
train_orders_df['product_embeddings'] = train_orders_df['product_id'].map(product_id_to_embedding)

# Discarding missing products
train_orders_df.dropna(inplace = True)

# Now we set both embeddings and ensure they are of the same size
train_order_embeddings = np.array(train_orders_df['orders_embeddings'].tolist())
train_product_embeddings = np.array(train_orders_df['product_embeddings'].tolist())
print(f'Training orders embeddings: {len(train_order_embeddings)}')
print(f'Training product embeddings: {len(train_product_embeddings)}')

# Setting emnbeddings as tensors
train_order_embeddings_ds = tf.data.Dataset.from_tensor_slices({"order_embedding": train_order_embeddings})
train_product_embeddings_ds = tf.data.Dataset.from_tensor_slices({"product_embedding": train_product_embeddings})

# Shuffling all trainin examples
dataset_size = len(train_order_embeddings)
train_ds = tf.data.Dataset.zip((train_order_embeddings_ds, train_product_embeddings_ds))
train_ds = train_ds.shuffle(buffer_size=dataset_size, seed=42)

# Defining train dataset in a suitable tensorflow format
train_ds = train_ds.map(lambda x, y: {**x, **y}).batch(32)

# Specifying embeddings dimension
embedding_dim = 768

# Initializing model
model = TwoTowerModel(embedding_dim, product_embeddings)

# Ensure the model is built by calling it on a batch of data (dummy step to ensure model to be built)
dummy_order_embedding = tf.constant(train_order_embeddings[:64])
dummy_product_embedding = tf.constant(train_product_embeddings[:64])
model.query_model({"order_embedding": dummy_order_embedding})
model.candidate_model({"product_embedding": dummy_product_embedding})

model.build(input_shape={"order_embedding": tf.TensorShape([None, embedding_dim]),
                         "product_embedding": tf.TensorShape([None, embedding_dim])})


# Model compilation with chosen optimizer
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.05))

# Defining epochs
num_epochs = 200

# Defining callbacks
checkpoint_filepath = 'models/checkpoint.model.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_weights_only=True,
    save_best_only=True,
    verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=2,
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)

print('Training....')
# Training process
try:
    history = model.fit(train_ds,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=[model_checkpoint_callback, early_stopping])
except Exception as e:
    print(f"An error occurred: {e}")
    # Save the model state on error
    model.save_weights(checkpoint_filepath)

print('Saving model...')
# Save the model after training
inputs = {
    "order_embedding": tf.keras.Input(shape=(embedding_dim,)),
    "product_embedding": tf.keras.Input(shape=(embedding_dim,))
}
model._set_inputs(inputs)
model.save('models/two_tower_architecture_trained')

print('Two-tower architecture successfully trained')