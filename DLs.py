# -*- coding: utf-8 -*-
"""
@author: andres.sanchez
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encoded_1 = layers.Dense(input_dim, activation='relu')
        self.encoded_2 = layers.Dense(128, activation='relu')
        self.encoded_3 = layers.Dense(64, activation='relu')
        self.latent_space = layers.Dense(32, activation = 'relu')
    
    def call(self, inputs):

        encoded_1 = self.encoded_1(inputs)
        encoded_2 = self.encoded_2(encoded_1)
        encoded_3 = self.encoded_3(encoded_2)
        latent_space = self.latent_space(encoded_3)
        
        return latent_space

class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        self.decoded_1 = layers.Dense(64, activation='relu')
        self.decoded_2 = layers.Dense(128, activation='relu')
        
        self.decoded_input_data = layers.Dense(input_dim, activation = 'linear')
        self.decoded_price = layers.Dense(1, activation = 'linear')
    
    def call(self, latent_space):
        
        decoded_1 = self.decoded_1(latent_space)
        decoded_2 = self.decoded_2(decoded_1)
        
        decoded_data = self.decoded_input_data(decoded_2)
        
        return decoded_data

class Autoencoder(tf.keras.Model):
    def __init__(self, input_data, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        
        self.input_dim = input_data.shape[1]
        
        self.data_loss_tracker = keras.metrics.Mean(name="data_loss")
        
        self.encoder = Encoder(self.input_dim)
        self.decoder = Decoder(self.input_dim)
        
    @property
    def metrics(self):
        return [
            self.data_loss_tracker,
        ]
    
    def call(self, inputs):
        latent_space = self.encoder(inputs)
        decoded_data = self.decoder(latent_space)
        return decoded_data, latent_space
    
    def train_step(self, data):
        x, y = data
        y = tf.cast(y, dtype=tf.float32)
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            latent_space = self.encoder(x)
            decoded_data = self.decoder(latent_space)
            
            data_loss = tf.reduce_mean(tf.math.squared_difference(x, decoded_data), 1)
        
        grads = tape.gradient(data_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.data_loss_tracker.update_state(data_loss)
        
        return {
                "data_loss": self.data_loss_tracker.result(),
                }
    
    def test_step(self, val_data):
        x, y = val_data
        
        y = tf.cast(y, dtype=tf.float32)
        x = tf.cast(x, dtype=tf.float32)
        
        latent_space = self.encoder(x)
        decoded_data = self.decoder(latent_space)
        
        data_loss = tf.reduce_mean(tf.math.squared_difference(x, decoded_data))
        
        return {
                "data_loss": data_loss,
                }
    
    def plot_latent_space(self, train_x, y):
        
        # price = df_train_imp.SalePrice
        price = y.to_numpy().ravel()
        price_normalized = (price - np.min(price)) / (np.max(price) - np.min(price))
        _, latent_embeddings = self.predict(train_x)
        tsne_embeddings = TSNE(n_components=2).fit_transform(latent_embeddings)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=price_normalized, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Normalized Price')
        plt.title('Latent Space Colored by Price')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()

class LossThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_data_loss")
        if loss is not None and loss < self.threshold:
            print(f"\nStopping training as loss has reached {loss:.4f}, below the threshold of {self.threshold}")
            self.model.stop_training = True

def plot_training_loss(history):

        plt.plot(history['data_loss'], label='Train Loss')
        plt.plot(history['val_data_loss'], label='Validation Loss')
        plt.title('Loss over training')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=1000,  
    decay_rate=0.8, 
    staircase=True  
)

        
