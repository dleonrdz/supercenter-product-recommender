import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

"""
This script defines the two-tower architecture suitable for receiving
pre-computed embeddings for training process and for inference.
"""

class OrderModel(tf.keras.Model):
    def __init__(self, embedding_dim):
        super().__init__()
        # First dense layer with ReLU activation
        self.dense_1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        # Dropout layer to prevent overfitting during training
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Second dense layer with ReLU activation
        self.dense_2 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        # Output layer to generate the final embedding
        self.output_layer = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):
        # Pass the input through the first dense layer
        x = self.dense_1(inputs["order_embedding"])
        # Apply dropout only during training
        x = self.dropout(x)
        # Pass the result through the second dense layer
        x = self.dense_2(x)
        # Generate the final output embedding
        return self.output_layer(x)

    def get_config(self):
        # Return the configuration of the model for serialization
        return {"embedding_dim": self.dense_1.units}


class ProductModel(tf.keras.Model):
    def __init__(self, embedding_dim):
        super().__init__()
        # First dense layer with ReLU activation
        self.dense_1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        # Dropout layer to prevent overfitting during training
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Second dense layer with ReLU activation
        self.dense_2 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        # Output layer to generate the final embedding
        self.output_layer = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):
        # Pass the input through the first dense layer
        x = self.dense_1(inputs["product_embedding"])
        # Apply dropout only during training
        x = self.dropout(x)
        # Pass the result through the second dense layer
        x = self.dense_2(x)
        # Generate the final output embedding
        return self.output_layer(x)

    def get_config(self):
        # Return the configuration of the model for serialization
        return {"embedding_dim": self.dense_1.units}


class TwoTowerModel(tfrs.models.Model):
    def __init__(self, embedding_dim, product_embeddings):
        super().__init__()
        # Define the query model for orders
        self.query_model = OrderModel(embedding_dim)
        # Define the candidate model for products
        self.candidate_model = ProductModel(embedding_dim)
        # Store the product embeddings
        self.product_embeddings = product_embeddings

        # Create a dataset from product embeddings for use in the retrieval task
        candidate_dataset = tf.data.Dataset.from_tensor_slices(product_embeddings).batch(128)
        candidate_dataset = candidate_dataset.map(lambda x: {"product_embedding": x})

        # Define the retrieval task with FactorizedTopK metric
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset,
            ),
        )

    def compute_loss(self, features, training=False):
        # Generate query embeddings from order embeddings
        query_embeddings = self.query_model({
            "order_embedding": features["order_embedding"]
        })
        # Generate candidate embeddings from product embeddings
        product_embeddings = self.candidate_model({
            "product_embedding": features["product_embedding"]
        })

        # Compute and return the retrieval task loss
        return self.task(
            query_embeddings, product_embeddings, compute_metrics=not training)

    def call(self, inputs):
        # Generate query embeddings from order embeddings
        query_embeddings = self.query_model({"order_embedding": inputs["order_embedding"]})
        # Generate candidate embeddings from product embeddings
        candidate_embeddings = self.candidate_model({"product_embedding": inputs["product_embedding"]})
        return query_embeddings, candidate_embeddings

    def get_config(self):
        # Return the configuration of the model for serialization
        return {
            "embedding_dim": self.query_model.dense_1.units,
            "product_embeddings": None,  # This cannot be serialized directly
        }

    @classmethod
    def from_config(cls, config):
        # Create an instance of the class from the configuration
        return cls(**config)

