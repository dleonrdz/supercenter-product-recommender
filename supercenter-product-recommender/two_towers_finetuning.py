import tensorflow as tf
class TwoTowerModel(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.product_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.order_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
    def call(self, inputs):
        product_embedding = self.product_tower(inputs['product_input'])
        order_embedding = self.order_tower(inputs['order_input'])
        return product_embedding, order_embedding


def train_step(model, product_embeddings, order_embeddings, optimizer):
    with tf.GradientTape() as tape:
        product_embeds, order_embeds = model({'product_input': product_embeddings, 'order_input': order_embeddings})
        loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(product_embeds, order_embeds, axis=1))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



