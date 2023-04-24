from tensorflow import keras
import tensorflow as tf
from vit import ViT
import os
import numpy as np
import tensorflow_addons as tfa

# Vision Transformer Keras Implementation:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/

# unsupervised pre-training: https://arxiv.org/pdf/2103.16554.pdf

print(tf.config.list_physical_devices('GPU'))

VGG_DATA_PATH = "/home/jovyan/VGGFace2-aligned/test"
input_size = (128, 128, 3)

#train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
#test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2-aligned/train")][1:]
test_classes = [i[0][-7:] for i in os.walk(VGG_DATA_PATH)][1:]
num_train_classes = len(train_classes)
num_test_classes = len(test_classes)

pre_model = ViT(
    input_size, 
    num_train_classes,
    projection_dim=64,
    mlp_head_units=[2048],
    transformer_units=[128, 64],
    patch_size=10,
    patch_stride=8
)

pre_model.model.load_weights("/home/jovyan/499_project/vit_checkpoint_pretrained")

pre_model.summary()

pre_model.model.layers.pop()
x = keras.layers.Dense(num_test_classes)(pre_model.model.layers[-2].output)

vit_model = keras.Model(inputs=pre_model.model.inputs, outputs=[x])
vit_model.summary()


df_train = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.25, seed=42, subset='training')
df_val = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.25, seed=42, subset='validation')

#history = vit_model.train(df_train, df_val, checkpoint_filepath="/home/jovyan/499_project/vit_checkpoint_pretrained_2", log_filepath="./train_log_pretrained.csv", epochs=300)

optimizer = tfa.optimizers.AdamW(
    learning_rate = 0.001, weight_decay = 0.0001,
)

vit_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/vit_checkpoint_pretrained_2",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

logger_callback = keras.callbacks.CSVLogger("./train_log_pretrained.csv", separator=",", append=False)

history = vit_model.fit(
    df_train,
    batch_size=256,
    epochs=300,
    validation_data=df_val,
    callbacks=[checkpoint_callback, stopping_callback, logger_callback],
)

# vit_model.evaluate(df_val, checkpoint_filepath="/home/jovyan/499_project/vit_checkpoint_pretrained")
