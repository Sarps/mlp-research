from keras import layers, models


class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq = models.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        return self.layer_norm(x)
