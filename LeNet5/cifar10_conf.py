from keras.datasets import cifar10
from comm import ConvLayer
import numpy as np
class Conf:
    def __init__(self):

        # structure
        self.image_shape  = (32,32,3)
        self.classes      = 10
        self.conv_layers  = [ConvLayer((5,5,3,16), (1,1,1,1), (1,2,2,1), (1,2,2,1)),
                             ConvLayer((5,5,16,32),(1,1,1,1),(1,2,2,1),(1,2,2,1))]
        self.fc_layers    = [256, self.classes]

        #dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape((-1,)+self.image_shape) / 255.0
        self.x_test = x_test.reshape((-1,)+self.image_shape) / 255.0
        y_train = (y_train.reshape(y_train.shape[0], -1) == np.arange(10)).astype(np.float32)
        self.y_test = (y_test.reshape(y_test.shape[0], -1) == np.arange(10)).astype(np.float32)
        training_size = int(0.9 * x_train.shape[0])
        self.x_valid = x_train[training_size:]
        self.y_valid = y_train[training_size:]
        self.x_train = x_train[0:training_size]
        self.y_train = y_train[0:training_size]

        #parameter
        self.batch_size = 16
        self.learning_rate_base = 0.01
        self.learning_rate_decay = 0.99
        self.regularization_rate = 0.00001
        self.training_epochs = 10000
        self.moving_average_decay = 0.99
        self.model_savepath = "/tmp/tf/cifar10"
        self.model_name = "cifar10.ckpt"


