
from archive.lib.configs.tranformers import TransformerModelConfigs
from mltu.tensorflow.transformer.layers import Encoder, Decoder
from keras import layers, Model
import tensorflow as tf


def Transformer(configs: TransformerModelConfigs, input_vocab_size: int, target_vocab_size: int) -> Model:
    encoder_input = layers.Input(shape=(configs.input_max_timesteps,), dtype=tf.uint16)
    decoder_input = layers.Input(shape=(configs.target_max_timesteps,), dtype=tf.uint16)

    encoder = Encoder(configs.num_layers, configs.d_model, configs.num_heads, configs.dff, input_vocab_size,
                      configs.dropout_rate)
    decoder = Decoder(configs.num_layers, configs.d_model, configs.num_heads, configs.dff, target_vocab_size,
                      configs.dropout_rate)
    dense = layers.Dense(target_vocab_size)

    encoder_tensor = encoder(encoder_input)
    decoder_tensor = decoder(decoder_input, encoder_tensor)
    output_tensor = dense(decoder_tensor)

    return Model(inputs=[encoder_input, decoder_input], outputs=output_tensor)
