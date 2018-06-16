import numpy as np
import keras
import keras.backend as BK

from keras.datasets import mnist

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
if BK.image_data_format() == "channels_first":
    X_train = X_train[:,np.newaxis,:,:].astype(np.float32)/255.0
    X_test = X_test[:,np.newaxis,:,:].astype(np.float32)/255.0
else:
    X_train = X_train[:,:,:,np.newaxis].astype(np.float32)/255.0
    X_test = X_test[:,:,:,np.newaxis].astype(np.float32)/255.0

Y_train = (Y_train[:,np.newaxis] == np.arange(10)).astype(np.float32)
Y_test = (Y_test[:,np.newaxis] == np.arange(10)).astype(np.float32)
image_shape = X_train.shape[1:]

print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
print(image_shape)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=[5,5],padding="SAME",activation="relu",input_shape=image_shape))
model.add(keras.layers.MaxPooling2D(pool_size=[2,2]))
model.add(keras.layers.Conv2D(64,kernel_size=[5,5],padding="SAME",activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=[2,2]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=500, activation="relu"))
model.add(keras.layers.Dense(units=10,activation="softmax"))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.categorical_crossentropy,
    metrics=["accuracy"]
)

model.fit(x=X_train, y=Y_train, batch_size=64, epochs=1,validation_data=(X_test,Y_test))

score = model.evaluate(X_test,Y_test)
print("loss,acc:",score[0],score[1])