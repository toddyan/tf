class ConvLayer:
    def __init__(self, conv_shape, conv_strides, pool_shape, pool_strides):
        self.conv_shape   = conv_shape
        self.conv_strides = conv_strides
        self.pool_shape   = pool_shape
        self.pool_strides = pool_strides
