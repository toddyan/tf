import keras
from keras.preprocessing import sequence
import numpy as np

from keras.datasets import imdb

maxlen = 80
batch_size = 32
max_word = 20000

(X_train,Y_train),(X_test,Y_test) = imdb.load_data(num_words=max_word)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
print(X_train[0],Y_train[0])

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=max_word, output_dim=128))
model.add(keras.layers.LSTM(units=128, activation='tanh', dropout=0.2, recurrent_dropout=0.2,))
model.add(keras.layers.Dense(units=1,activation='sigmoid'))
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.binary_crossentropy,
    metrics=['accuracy']
)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=5, validation_data=(X_test,Y_test))
score = model.evaluate(X_test, Y_test, batch_size)
print("loss,acc",score[0],score[1])