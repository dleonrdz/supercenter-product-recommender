import tensorflow as tf
class TwoTowerModel(tf.keras.Model):
    def __init__(self, product_embedding_dim, order_embedding_dim):
        super(TwoTowerModel, self).__init__()
        self.product_embedding_layer = tf.keras.layers.Dense(product_embedding_dim, activation='relu')
        self.order_embedding_layer = tf.keras.layers.Dense(order_embedding_dim, activation='relu')
    def call(self, inputs):
        product_embedding = self.product_embedding_layer(inputs['product_input'])
        order_embedding = self.order_embedding_layer(inputs['order_input'])
        return product_embedding, order_embedding

def train_step(model, product_embeddings, order_embeddings, optimizer):
    with tf.GradientTape() as tape:
        product_embeds, order_embeds = model({'product_input': product_embeddings, 'order_input': order_embeddings})
        loss = tf.reduce_mean(tf.losses.cosine_similarity(product_embeds, order_embeds))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



