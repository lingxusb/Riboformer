# main model structure

import tensorflow as tf
from tensorflow import keras
from keras import layers

from modules import ConvTower, TransformerBlock, TokenAndPositionEmbedding

class RiboSTD(keras.Model):

    def __init__(self, configs):
        super().__init__()
        
        self.wsize = configs.wsize
        self.embed_dim = configs.embed_dim
        
        self.embedding_layer = TokenAndPositionEmbedding(configs.wsize, configs.vocab_size, configs.embed_dim)
        
        self.conv_tower1 = ConvTower('2D', [32,32,32,32,32], 5, activation = configs.activation)
        self.conv_tower2 = ConvTower('1D', [32,32,32,32,8], 9, activation = configs.activation)
        
        self.transformer_block1 = TransformerBlock(configs.embed_dim, configs.num_heads, configs.mlp_dim)
        self.transformer_block2 = TransformerBlock(configs.embed_dim, configs.num_heads, configs.mlp_dim)
        
        self.head1 = keras.Sequential([layers.Flatten(),
                                       layers.Dropout(configs.dropout_rate),
                                       layers.Dense(32, activation = configs.activation)])
        self.head2 = keras.Sequential([layers.Flatten(),
                                       layers.Dropout(configs.dropout_rate),
                                       layers.Dense(32, activation = configs.activation)])
        
        self.final_dense = layers.Dense(1, activation = configs.activation, name = "read_depth")
    
    
    def call(self, inputs):
        seq, exp = inputs

        # mRNA sequence branch
        x = self.embedding_layer(seq)
        x = tf.reshape(x, (-1, self.wsize, self.embed_dim, 1))
        
        x = self.conv_tower1(x)
        x = tf.reduce_mean(x, axis=-1)
        
        x, weights = self.transformer_block1(x)
        x = self.head1(x)
        
        # control experiment branch
        y = tf.reshape(exp, (-1, self.wsize, 1))
        
        y = self.conv_tower2(y)

        y, weights = self.transformer_block2(y)
        y = self.head2(y)

        outputs = self.final_dense(x * y)
        
        return outputs