# src/lstm_model.py

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm():

    model = models.Sequential()

    model.add(layers.Input(shape=(24, 1)))
    model.add(layers.LSTM(50))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse"
    )

    return model