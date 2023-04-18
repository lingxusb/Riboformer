# modules for the model

import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten


class ConvTower(layers.Layer):
    def __init__(self, 
                 func: str, 
                 filters: list[int],
                 kernel_size: int,
                 padding: str = "same", 
                 activation: str = None):
        super(ConvTower, self).__init__()
        
        self.func = layers.Conv1D if func == '1D' else layers.Conv2D
        
        self.nn = keras.Sequential()
        for f in filters:
            self.nn.add(self._create_conv_block(f, kernel_size, padding, activation))
            
    def call(self, inputs):
        output = self.nn(inputs)
        return output
    
    # stack of 2D convolutional layers with batch normalization
    def _create_conv_block(self, filters: int, kernel_size: int, padding: str, activation : str):
        block = [self.func(filters, kernel_size, padding = padding, activation = activation),
                 layers.BatchNormalization()]
        return keras.Sequential(block)


# transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 ff_dim, 
                 dropout_rate = 0.1,
                 **kwargs,):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.mha = layers.MultiHeadAttention(num_heads = self.num_heads, 
                                             key_dim = self.embed_dim)
        self.mlp = keras.Sequential([layers.Dense(self.ff_dim, activation = "relu"), 
                                     layers.Dense(self.embed_dim),])
        
        self.layernorm = layers.LayerNormalization(epsilon = 1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training: bool):
        
        # multi-head attention layer
        mha_output, weights = self.mha(inputs, inputs, return_attention_scores = True)
        mha_output = self.dropout(mha_output, training = training)
        mha_output += inputs
        
        # multi-layer perceptron layer
        output = self.layernorm(mha_output)
        output = self.mlp(output)
        
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.ff_dim,
        })
        return config


# token and position encoding functions
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, 
                 maxlen, 
                 vocab_size, 
                 embed_dim,
                 **kwargs,):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim = self.vocab_size, 
                                          output_dim = self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim = self.maxlen, 
                                        output_dim = self.embed_dim)

    def call(self, inputs):
        positions = self.pos_emb(tf.range(start = 0, limit = self.maxlen))
        outputs = self.token_emb(inputs)
        
        return outputs + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_len": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config
