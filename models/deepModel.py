# Made by Sarvesh Bhatnagar

from tensorflow import keras
from tensorflow.keras import layers


class NNModels:
    def __init__(self, input_shape=(100,), output_shape=2):
        """
        input shape example: (100, )
        output shape example: 2

        Each dl model will have dl_x and dl_x_compile functions
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def dl_0(self):
        inputs = keras.Input(shape=self.input_shape, name="digits")
        x = layers.Dense(64, activation=keras.layers.LeakyReLU(
            alpha=0.01), name="dense_1")(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.01)(x)
        x = layers.Dense(32, activation=keras.layers.LeakyReLU(
            alpha=0.01), name="dense_3")(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Dense(32, activation=keras.layers.LeakyReLU(
            alpha=0.01), name="dense_4")(x)
        x = layers.Dropout(0.05)(x)
        z = layers.Dense(12, activation="relu", name="dense_5")(x)
        y = layers.Dense(6, activation="relu", name="dense_6")(x)
        x = layers.Concatenate()([z, y])
        outputs = layers.Dense(
            self.output_shape, activation="softmax", name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def dl_0_compile(self, model):
        """
        Compile model dl_0
        """
        model.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
