class Config(object):
    
    def __init__(self,
                 wsize: int = 40,
                 vocab_size: int = 64,
                 embed_dim: int = 8,
                 mlp_dim: int = 64,
                 num_heads: int = 10,
                 dropout_rate: float = 0.4,
                 activation: str = 'relu'):
        super().__init__()
        
        self.wsize = wsize                  # sequence length
        self.vocab_size = vocab_size        # vocab for the transformer layer
        self.embed_dim = embed_dim          # embedding size for each token
        self.num_heads = num_heads          # number of attention heads
        self.mlp_dim = mlp_dim              # hidden layer size in the mlp
        self.dropout_rate = dropout_rate    # dropout rate
        self.activation = activation        # activation function used
        
    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, str(value)) for key, value in self.__dict__.items()])