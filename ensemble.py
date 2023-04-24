import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import convnext
from vit import ViT
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class WeightedAverageLayer(tf.keras.layers.Layer):
    def __init__(self, w1, w2, **kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)
        self.w1 = w1
        self.w2 = w2

    def call(self, inputs):
        return self.w1 * inputs[0] + self.w2 * inputs[1]

print(tf.config.list_physical_devices('GPU'))

VGG_DATA_PATH = "/home/jovyan/VGGFace2-aligned/"
input_size = (128, 128, 3)

#train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
#test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
# train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/train/")][1:]
test_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2-aligned/")][1:]
num_classes = len(test_classes)
df_train = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='training')
df_val = keras.preprocessing.image_dataset_from_directory(VGG_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='validation')

vit_model = ViT(input_size, num_classes, patch_size=10, patch_stride=8)
vit_model.model.load_weights("/home/jovyan/499_project/vit_checkpoint_aligned")
for layer in vit_model.model.layers:
    layer.trainable=False
vit_model.summary()

convnext_model = convnext.create_model(input_shape=(128,128), pretrained=False, num_classes=num_classes, include_top=True)
convnext_model.load_weights("/home/jovyan/499_project/convnext_checkpoint_aligned")
for layer in convnext_model.layers:
    layer.trainable=False
convnext_model.summary()

model_inputs = keras.layers.Input(input_size)
model_outputs = [vit_model.model(model_inputs), convnext_model(model_inputs)]
ensemble_outputs = WeightedAverageLayer(0.5, 0.5)(model_outputs)
ensemble_model = keras.Model(inputs=model_inputs, outputs=ensemble_outputs)

ensemble_model.summary()

optimizer = tfa.optimizers.AdamW(
    learning_rate=4e-3, weight_decay=0.05
)

ensemble_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
],)

'''checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/ensemble_checkpoint",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)


ensemble_model.fit(
    df_train,
    batch_size=256,
    epochs=300,
    validation_data=df_val,
    callbacks=[checkpoint_callback, stopping_callback]
)'''

#ensemble_model.evaluate(df_train)

predictions = np.array([])
labels =  np.array([])
for x, y in df_val:
    predictions = np.concatenate([predictions, np.argmax(ensemble_model.predict(x), axis=-1)])
    labels = np.concatenate([labels, y.numpy()])

conf_mat = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()[:50,:50]
print(conf_mat)
plt.matshow(conf_mat, cmap='binary')
plt.savefig('conf_mat_ensemble.png')