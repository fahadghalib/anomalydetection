"""Shared components of the neural models (manuscript Sections 3.3-3.5).

  - AdditiveAttention: the attention layer of Equations (7)-(8)
  - make_weighted_focal_loss: class-weighted focal loss, Equation (10)
  - sqrt_scaled_class_weights: class weights, Equation (11)
  - make_optimizer: AdamW with cosine annealing and warm restarts

The proposed network itself is assembled in
src/run_corrected.build_cnn_bigru_attention.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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


def make_weighted_focal_loss(class_weights: np.ndarray, gamma: float = config.FOCAL_GAMMA,
                              label_smoothing: float = config.LABEL_SMOOTHING):
    """Weighted Categorical Focal Loss, Eq. (7): FL(p_c) = -alpha_c (1-p_c)^gamma log(p_c).

    class_weights: array of length n_classes, the per-class alpha_c term
    (square-root-scaled inverse class frequency, computed from the training
    fold's post-resampling distribution in run_corrected._fit_keras).
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
