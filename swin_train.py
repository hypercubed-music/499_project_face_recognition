import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from swin_transformer import SwinTransformer

VGG_DATA_PATH = "/home/jovyan/VGGFace2/VGG-Face2/data/"
input_shape = (128, 128, 3)

train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/train/")][1:]
test_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/test/")][1:]
num_classes = 500
learning_rate = 5e-4
weight_decay = 0.05

swin_model = SwinTransformer(
    num_classes=500,
    img_size=128,
    drop_path_rate=0.2,
    window_size=8
)

#swin_model.build(input_shape)
swin_model.load_weights( "/home/jovyan/499_project/swin_checkpoint")

df_train = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH + "test/", image_size=input_shape[:2], validation_split=0.1, seed=42, subset='training', label_mode='categorical')
df_val = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH + "test/", image_size=input_shape[:2], validation_split=0.1, seed=42, subset='validation', label_mode='categorical')

optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

swin_model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
],)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/swin_checkpoint_2",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

#swin_model.summary()

history = swin_model.fit(
    df_train,
    batch_size=256,
    epochs=100,
    validation_data=df_val,
    callbacks=[checkpoint_callback],
)