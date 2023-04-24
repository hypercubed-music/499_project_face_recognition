import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import convnext
import pickle

VGG_DATA_PATH = "/home/jovyan/VGGFace2-aligned/"
input_size = (128, 128, 3)

#train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
#test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/train/")][1:]
test_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2-aligned/")][1:]
num_pretrained_classes = len(train_classes)
num_classes = len(test_classes)

base_model = convnext.create_model(input_shape=(128,128), pretrained=False, num_classes=num_classes, include_top=True)
# base_model.load_weights("/home/jovyan/499_project/convnext_checkpoint")
base_model.summary()

df_train = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='training')
df_val = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='validation')

optimizer = tfa.optimizers.AdamW(
    learning_rate=4e-3, weight_decay=0.05
)

base_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
],)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/convnext_checkpoint_aligned",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = base_model.fit(
    df_train,
    batch_size=256,
    epochs=300,
    validation_data=df_val,
    callbacks=[checkpoint_callback, stopping_callback],
)

with open('/home/jovyan/499_project/convnext_small_history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)