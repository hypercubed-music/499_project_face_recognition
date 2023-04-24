import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import convnext

VGG_DATA_PATH = "/home/jovyan/VGGFace2/VGG-Face2/data/"
input_size = (128, 128, 3)

train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/train/")][1:]
test_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/test/")][1:]
num_classes = len(train_classes)
print("Number of classes: " + str(num_classes))

base_model = convnext.create_model(input_shape=(128,128), pretrained=False, num_classes=num_classes)
base_model.load_weights("/home/jovyan/499_project/convnext_pretrained_checkpoint")
base_model.summary()

df_train = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH + "train/", image_size=input_size[:2], validation_split=0.1, seed=42, subset='training')
df_val = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH + "train/", image_size=input_size[:2], validation_split=0.1, seed=42, subset='validation')

optimizer = tfa.optimizers.AdamW(
    learning_rate=0.001, weight_decay=0.0001
)

base_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
],)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/convnext_pretrained_checkpoint",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

history = base_model.fit(
    df_train,
    batch_size=256,
    epochs=15,
    validation_data=df_val,
    callbacks=[checkpoint_callback],
)