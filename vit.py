import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

class Patches(layers.Layer):
    def __init__(self, patch_size, patch_stride):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        assert patch_stride <= patch_size, "Patch stride should be less than or equal to patch size"

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_stride, self.patch_stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, patches.shape[1] ** 2, patches.shape[3]])
        return patches



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class ViT():
    # Vision Transformer
    # Modified to allow overlapping patches
    # as per https://arxiv.org/abs/2103.14803
    def __init__(self, input_shape, num_classes, projection_dim=64, num_heads=8, transformer_units=[128, 64] ,transformer_layers = 20, mlp_head_units = [2048], patch_size=6, patch_stride=3):
        num_patches = ((input_shape[0] - (patch_size - patch_stride)) // patch_stride) ** 2

        inputs = layers.Input(shape=input_shape)
        # Create patches.
        patches = Patches(patch_size, patch_stride)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        self.model = keras.Model(inputs=inputs, outputs=logits)

    def summary(self):
        self.model.summary()

    def train(self, df, df_val, batch_size=256, epochs=100, learning_rate = 0.001, 
            weight_decay = 0.0001, checkpoint_filepath="/tmp/checkpoint", log_filepath="./train_log.csv"):
        print(df)
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        
        stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        logger_callback = keras.callbacks.CSVLogger(filename, separator=",", append=False)

        history = self.model.fit(
            df,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=df_val,
            callbacks=[checkpoint_callback, stopping_callback, logger_callback],
        )

        self.model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(df)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history

    def evaluate(self, df, learning_rate = 0.001, weight_decay = 0.0001, checkpoint_filepath="/tmp/checkpoint"):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        self.model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(df)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        