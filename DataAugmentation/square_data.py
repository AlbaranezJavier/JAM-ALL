from tensorflow.keras import layers

class SquareAugmentation(layers.Layer):
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.flip = tfi.random

    def call(self, x):
        pass
