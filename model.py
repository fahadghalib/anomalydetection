"""Multi-scale 1D-CNN + BiGRU + Additive Attention hybrid model (Sec 3.3).

Implements Eq. (1)-(8) of the paper:
  - parallel 1D-CNN branches with kernel sizes K=1,3,5 (Eq. 1)
  - Bidirectional GRU (Eq. 2-5)
  - additive attention producing a context vector (Eq. 6)
  - softmax readout over 20 classes (Eq. 8)
  - Weighted Categorical Focal Loss with sqrt-scaled class weights (Eq. 7)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from . import config


@tf.keras.utils.register_keras_serializable(package="anomaly")
class AdditiveAttention(layers.Layer):
    """Bahdanau-style additive attention over the time (feature-sequence) axis.

    score_i = tanh(W h_i + b) -> scalar per position
    alpha   = softmax(score)                (Eq. 6 weights)
    z       = sum_i alpha_i * h_i           (Eq. 6 context vector)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, activation="tanh", name="attn_score")

    def build(self, input_shape):
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, D)
        score = self.score_dense(inputs)          # (batch, T, 1)
        score = tf.squeeze(score, axis=-1)         # (batch, T)
        alpha = tf.nn.softmax(score, axis=-1)      # (batch, T)
        alpha = tf.expand_dims(alpha, axis=-1)     # (batch, T, 1)
        context = tf.reduce_sum(inputs * alpha, axis=1)  # (batch, D)
        return context, alpha

    def get_config(self):
        return super().get_config()


def build_model(
    n_features: int = config.N_FEATURES,
    n_classes: int = config.N_CLASSES,
) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features, 1), name="features")

    branches = []
    for k in config.CNN_KERNELS:
        b = layers.Conv1D(
            config.CNN_FILTERS, k, padding="same", activation="relu",
            name=f"conv1d_k{k}",
        )(inp)
        b = layers.BatchNormalization()(b)
        branches.append(b)
    x = layers.Concatenate(name="multiscale_concat")(branches)

    x = layers.Bidirectional(
        layers.GRU(
            config.GRU_UNITS,
            return_sequences=True,
            recurrent_dropout=config.RECURRENT_DROPOUT,
        ),
        name="bigru",
    )(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="post_gru_dropout")(x)

    context, _alpha = AdditiveAttention(name="additive_attention")(x)

    out = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(config.L2_REG),
        name="readout",
    )(context)

    return models.Model(inp, out, name="cnn_bigru_attention")


def make_weighted_focal_loss(class_weights: np.ndarray, gamma: float = config.FOCAL_GAMMA,
                              label_smoothing: float = config.LABEL_SMOOTHING):
    """Weighted Categorical Focal Loss, Eq. (7): FL(p_c) = -alpha_c (1-p_c)^gamma log(p_c).

    class_weights: array of length n_classes, the per-class alpha_c term
    (square-root-scaled inverse class frequency, computed from the training
    fold's *post-resampling* distribution -- see train.py).
    """
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    n_classes = class_weights.shape[0]

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n_classes

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_c = tf.reduce_sum(class_weights * y_true, axis=-1)
        pc = tf.reduce_sum(y_true * y_pred, axis=-1)
        ce = -tf.math.log(pc)
        loss = alpha_c * tf.pow(1.0 - pc, gamma) * ce
        return loss

    return loss_fn


def make_optimizer(steps_per_epoch: int):
    first_decay_steps = max(1, steps_per_epoch * config.COSINE_FIRST_DECAY_STEPS_EPOCHS)
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.INITIAL_LR,
        first_decay_steps=first_decay_steps,
    )
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=config.WEIGHT_DECAY,
    )


def sqrt_scaled_class_weights(class_counts: dict[int, int], n_classes: int = config.N_CLASSES) -> np.ndarray:
    """alpha_c ~ 1/sqrt(freq_c), normalized to mean 1 across classes present."""
    counts = np.array([class_counts.get(c, 1) for c in range(n_classes)], dtype="float64")
    counts = np.maximum(counts, 1)
    inv_sqrt = 1.0 / np.sqrt(counts)
    alpha = inv_sqrt / inv_sqrt.mean()
    return alpha.astype("float32")
