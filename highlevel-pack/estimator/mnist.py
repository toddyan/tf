import keras
import numpy as np

from keras.datasets import mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0],-1)).astype(np.float32)/255.0
X_test = X_test.reshape((X_test.shape[0],-1)).astype(np.float32)/255.0
Y_train = (Y_train[:,np.newaxis] == np.arange(10)).astype(np.float32)
Y_test = (Y_test[:,np.newaxis] == np.arange(10)).astype(np.float32)
image_shape = X_train.shape[1:]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, image_shape)

