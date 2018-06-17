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

input1 = keras.layers.Input(shape=image_shape, name="input1")
input2 = keras.layers.Input(shape=(10,), name="input2")
hidden1 = keras.layers.Dense(units=5,activation='relu')(input1)
output1 = keras.layers.Dense(units=10,activation='softmax', name='output1')(hidden1)
concat = keras.layers.concatenate(inputs=[hidden1,input2])
output2 = keras.layers.Dense(units=10, activation='softmax', name='output2')(concat)

model = keras.Model(inputs=[input1,input2], outputs=[output1,output2])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={
        'output1':keras.losses.categorical_crossentropy,
        'output2':keras.losses.categorical_crossentropy
    },
    loss_weights=[1.0,0.5],
    metrics=['accuracy']
)

model.fit(
    x=[X_train,Y_train],
    y=[Y_train, Y_train],
    batch_size=64,
    epochs=10,
    validation_data=([X_test, Y_test],[Y_test,Y_test])
)

score = model.evaluate(x=[X_test, Y_test],y=[Y_test,Y_test])

print(score)
