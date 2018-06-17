import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0],-1)).astype(np.float32)/255.0
X_test = X_test.reshape((X_test.shape[0],-1)).astype(np.float32)/255.0
Y_train = (Y_train[:,np.newaxis] == np.arange(10)).astype(np.float32)
Y_test = (Y_test[:,np.newaxis] == np.arange(10)).astype(np.float32)
image_shape = X_train.shape[1:]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, image_shape)

X = tf.placeholder(dtype=tf.float32, shape=(None,784), name="feature")
Y = tf.placeholder(dtype=tf.float32, shape=(None,10), name="label")

hidden = tf.keras.layers.Dense(units=500,activation='relu')(X)
Y_pred = tf.keras.layers.Dense(units=10, activation='softmax')(hidden)

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=Y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true=Y, y_pred=Y_pred))

with tf.Session() as s:
    tf.global_variables_initializer().run()
    for epoch in range(1,10):
        print("epoch %d:" % epoch)
        start = 0
        while start < X_train.shape[0]:
            end = min(start+64, X_train.shape[0])
            _, lo, acc = s.run([optimizer, loss, accuracy], feed_dict={X:X_train[start:end], Y:Y_train[start:end]})
            if(start/64%100==1): print(lo,acc)
            start = end
    print(s.run(accuracy, feed_dict={X:X_test,Y:Y_test}))