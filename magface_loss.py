import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

# From https://github.com/leondgarse/Keras_insightface/tree/12cba0c8837da5e47032cc283f4aa2c1a783f5ac

class ArcfaceLossSimple(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossSimple, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        # self.low_pred_punish = tf.sin(np.pi - margin) * margin
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLossSimple, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MagFaceLoss(ArcfaceLossSimple):
    """ Another set for fine-tune is: min_feature_norm, max_feature_norm, min_margin, max_margin, regularizer_loss_lambda = 1, 51, 0.45, 1, 5 """

    def __init__(
        self,
        min_feature_norm=10.0,
        max_feature_norm=110.0,
        min_margin=0.45,
        max_margin=0.8,
        scale=64.0,
        regularizer_loss_lambda=35.0,
        use_cosface_margin=False,
        curricular_hard_scale=-1,
        from_logits=True,
        label_smoothing=0,
        **kwargs
    ):
        super(MagFaceLoss, self).__init__(scale=scale, from_logits=from_logits, label_smoothing=label_smoothing, **kwargs)
        # l_a, u_a, lambda_g
        self.min_feature_norm, self.max_feature_norm, self.regularizer_loss_lambda = min_feature_norm, max_feature_norm, regularizer_loss_lambda
        # l_margin, u_margin
        self.min_margin, self.max_margin = min_margin, max_margin
        self.use_cosface_margin, self.curricular_hard_scale = use_cosface_margin, curricular_hard_scale
        self.margin_scale = (max_margin - min_margin) / (max_feature_norm - min_feature_norm)
        self.regularizer_loss_scale = 1.0 / (self.max_feature_norm ** 2)
        self.use_curricular_scale = False
        self.epislon = 1e-3
        if curricular_hard_scale >= 0:
            self.curricular_hard_scale = tf.Variable(curricular_hard_scale, dtype="float32", trainable=False)
            self.use_curricular_scale = True
        # np.set_printoptions(precision=4)
        # self.precission_4 = lambda xx: tf.math.round(xx * 10000) / 10000

    def call(self, y_true, norm_logits_with_norm):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # feature_norm is multiplied with -1 in NormDense layer, keeping low for not affecting accuracy metrics.
        norm_logits, feature_norm = norm_logits_with_norm[:, :-1], norm_logits_with_norm[:, -1] * -1
        norm_logits = tf.clip_by_value(norm_logits, -1 + self.epislon, 1 - self.epislon)
        feature_norm = tf.clip_by_value(feature_norm, self.min_feature_norm, self.max_feature_norm)
        # margin = (self.u_margin-self.l_margin) / (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        margin = self.margin_scale * (feature_norm - self.min_feature_norm) + self.min_margin
        margin = tf.expand_dims(margin, 1)

        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        if self.use_cosface_margin:
            # Cosface process
            arcface_logits = tf.where(tf.cast(y_true, dtype=tf.bool), norm_logits - margin, norm_logits) * self.scale
            # theta_valid = y_pred_vals - margin
        else:
            # Arcface process
            margin_cos, margin_sin = tf.cos(margin), tf.sin(margin)
            # XLA after TF > 2.7.0 not supporting this gather_nd -> tensor_scatter_nd_update method...
            # threshold = tf.cos(np.pi - margin)
            # theta = y_pred_vals * margin_cos - tf.sqrt(tf.maximum(1 - tf.pow(y_pred_vals, 2), 0.0)) * margin_sin
            # theta_valid = tf.where(y_pred_vals > threshold, theta, self.theta_min - theta)
            # arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
            arcface_logits = tf.where(
                tf.cast(y_true, dtype=tf.bool),
                norm_logits * margin_cos - tf.sqrt(tf.maximum(1 - tf.pow(norm_logits, 2), 0.0)) * margin_sin,
                norm_logits,
            )
            arcface_logits = tf.minimum(arcface_logits, norm_logits) * self.scale

        # if self.use_curricular_scale:
        #     self.curricular_hard_scale.assign(tf.reduce_mean(y_pred_vals) * 0.01 + (1 - 0.01) * self.curricular_hard_scale)
        #     tf.print(", hard_scale:", self.curricular_hard_scale, end="")
        #     norm_logits = tf.where(norm_logits > tf.expand_dims(theta_valid, 1), norm_logits * (self.curricular_hard_scale + norm_logits), norm_logits)

        arcface_loss = tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

        # MegFace loss_G, g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        regularizer_loss = self.regularizer_loss_scale * feature_norm + 1.0 / feature_norm

        tf.print(
            # ", regularizer_loss: ",
            # tf.reduce_mean(regularizer_loss),
            ", arcface: ",
            tf.reduce_mean(arcface_loss),
            ", margin: ",
            tf.reduce_mean(margin),
            # ", min: ",
            # tf.reduce_min(margin),
            # ", max: ",
            # tf.reduce_max(margin),
            ", feature_norm: ",
            tf.reduce_mean(feature_norm),
            # ", min: ",
            # tf.reduce_min(feature_norm),
            # ", max: ",
            # tf.reduce_max(feature_norm),
            sep="",
            end="\r",
        )
        return arcface_loss + regularizer_loss * self.regularizer_loss_lambda

    def get_config(self):
        config = super(MagFaceLoss, self).get_config()
        config.update(
            {
                "min_feature_norm": self.min_feature_norm,
                "max_feature_norm": self.max_feature_norm,
                "min_margin": self.min_margin,
                "max_margin": self.max_margin,
                "regularizer_loss_lambda": self.regularizer_loss_lambda,
                "use_cosface_margin": self.use_cosface_margin,
                "curricular_hard_scale": K.get_value(self.curricular_hard_scale),
            }
        )
        return config