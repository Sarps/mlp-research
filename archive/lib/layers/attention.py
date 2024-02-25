from keras import layers


class BaseAttention(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_vector, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_vector])
        return self.layernorm(x)


class SelfAttention(BaseAttention):
    def call(self, x):
        attn_vector = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_vector])
        return self.layernorm(x)


class MaskedSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        return self.layernorm(x)
