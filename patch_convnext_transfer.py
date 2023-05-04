import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import patch_convnet
import convnext

VIDEO_DATA_PATH = "/home/jovyan/aligned_images_DB"
input_size = (128, 128, 3)

#train_list = open(VGG_DATA_PATH + "train_list.txt", "r")
#test_list = open(VGG_DATA_PATH + "test_list.txt", "r")
#train_classes = [i[0][-7:] for i in os.walk("/home/jovyan/VGGFace2/VGG-Face2/data/train/")][1:]
test_classes = [i[0][-7:] for i in os.walk("/home/jovyan/aligned_images_DB")][1:]
num_classes = len(test_classes)

conv_stem = patch_convnet.build_convnext_stem(dims=[96, 192, 384, 768])
conv_trunk = patch_convnet.ConvNeXt_Trunk(depths=[3, 3, 27, 3], dimensions=[96, 192, 384, 768])

model = patch_convnet.PatchConvNet(
    stem=conv_stem,
    trunk=conv_trunk,
    attention_pooling=patch_convnet.AttentionPooling(dimensions=768, num_classes=500)
)
model(keras.layers.Input(shape=(224, 224, 3)))

new_model = patch_convnet.PatchConvNet(
    stem=conv_stem,
    trunk=conv_trunk,
    attention_pooling=patch_convnet.AttentionPooling(dimensions=768, num_classes=1595)
)
new_model(keras.layers.Input(shape=(224, 224, 3)))

from keras import backend as K

class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = self.lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)
        
    def lr_warmup_cosine_decay(self,global_step,
                               warmup_steps,
                               hold = 0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

model.summary()
model.load_weights("/home/jovyan/499_project/patch_conv_pretrained_checkpoint")

new_model.layers[0].set_weights(model.layers[0].get_weights())
new_model.layers[1].set_weights(model.layers[1].get_weights())
new_model.layers[0].trainable = False
new_model.layers[1].trainable = False
new_model.summary()

df_train = keras.preprocessing.image_dataset_from_directory(VIDEO_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='training', batch_size=256)
df_val = keras.preprocessing.image_dataset_from_directory(VIDEO_DATA_PATH, image_size=input_size[:2], validation_split=0.1, seed=42, subset='validation', batch_size=256)    
df_train = df_train.map(lambda x, y: (keras.layers.Resizing(224, 224)(x), y))
df_val = df_val.map(lambda x, y: (keras.layers.Resizing(224, 224)(x), y))

AUTOTUNE=tf.data.AUTOTUNE
df_train.cache().prefetch(buffer_size=AUTOTUNE)
df_val.cache().prefetch(buffer_size=AUTOTUNE)

optimizer = tfa.optimizers.AdamW(
    learning_rate=0.0004, weight_decay=0.00001
)

new_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
])

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "/home/jovyan/499_project/patch_conv_video_checkpoint",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)

stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

logger_callback = keras.callbacks.CSVLogger('./train_log_pcnn_video.csv', separator=",", append=False)

total_steps = len(df_train)*300

lr_callback = WarmupCosineDecay(total_steps=total_steps, warmup_steps=len(df_train), hold=len(df_train), start_lr=0.0, target_lr=0.0004)

history = new_model.fit(
    df_train,
    batch_size=256,
    epochs=300,
    validation_data=df_val,
    callbacks=[checkpoint_callback, stopping_callback, logger_callback, lr_callback],
)