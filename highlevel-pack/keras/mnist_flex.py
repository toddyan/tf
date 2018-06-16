import keras
import numpy as np

from keras.datasets import mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
if keras.backend.image_data_format() == "channels_last":
    X_train = X_train[:,:,:,np.newaxis].astype(np.float32)/255.0
    X_test = X_test[:,:,:,np.newaxis].astype(np.float32)/255.0
else:
    X_train = X_train[:,np.newaxis,:,:].astype(np.float32)/255.0
    X_test = X_test[:,np.newaxis,:,:].astype(np.float32)/255.0
Y_train = (Y_train[:,np.newaxis] == np.arange(10)).astype(np.float32)
Y_test = (Y_test[:,np.newaxis] == np.arange(10)).astype(np.float32)
image_shape = X_train.shape[1:]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

input = keras.layers.Input(shape=image_shape)
layer = keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='SAME', activation='relu')(input)
layer = keras.layers.MaxPooling2D(pool_size=[2,2])(layer)
layer = keras.layers.Conv2D(filters=64, kernel_size=[5,5], padding='SAME', activation='relu')(layer)
layer = keras.layers.MaxPooling2D(pool_size=[2,2])(layer)
layer = keras.layers.Flatten()(layer)
layer = keras.layers.Dense(units=500, activation='relu')(layer)
layer = keras.layers.Dense(units=10, activation='softmax')(layer)
model = keras.Model(inputs=input, outputs=layer)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
model.fit(X_train, Y_train, batch_size=64, epochs=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
print(score)