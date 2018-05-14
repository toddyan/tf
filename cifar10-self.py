import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
(raw_x,raw_y),(raw_x_test,raw_y_test) = cifar10.load_data()
X_train = raw_x.reshape(raw_x.shape[0],-1)/255.0
X_test  = raw_x_test.reshape(raw_x_test.shape[0],-1)/255.0
Y_train = (raw_y == np.arange(10)).astype(np.float32)
Y_test  = (raw_y_test == np.arange(10)).astype(np.float32)
TrainingSize = int(0.9 * X_train.shape[0])
X_valid = X_train[TrainingSize:]
Y_valid = Y_train[TrainingSize:]
X_train = X_train[0:TrainingSize]
Y_train = Y_train[0:TrainingSize]

print X_train.shape
print Y_train.shape
print X_valid.shape
print Y_valid.shape
print X_test.shape
print Y_test.shape

X = tf.placeholder(tf.float32,shape=(None,3072))
Y = tf.placeholder(tf.float32,shape=(None,10))
N = [3072, 800, 100, 10]
for i in range(1,len(N)):
    w = tf.Variable(tf.truncated_normal((N[i-1],N[i]),dtype=tf.float32,stddev=0.1))
    if i == 1:
        z = tf.matmul(X, w) + tf.Variable(tf.zeros(N[i],))
    else:
        z = tf.matmul(A, w) + tf.Variable(tf.zeros(N[i],))
    if i + 1 < len(N):
        A = tf.nn.relu(z)
    else:
        crossEnt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z,labels=Y))
optimizer = tf.train.AdamOptimizer(0.002).minimize(crossEnt)
batch_size = 64
with tf.Session() as s:
    tf.global_variables_initializer().run()
    for epoch in range(20):
        start = 0
        while start < X_train.shape[0]:
            end = min(X_train.shape[0], start + batch_size)
            s.run(optimizer,feed_dict={X:X_train[start:end],Y:Y_train[start:end]})
            start = end
        z_train_predict = s.run(z, feed_dict={X:X_train}).argmax(axis=1)
        print (z_train_predict == Y_train.argmax(axis=1)).astype(np.float32).mean()
        z_valid_predict = s.run(z, feed_dict={X:X_valid}).argmax(axis=1)
        print (z_valid_predict == Y_valid.argmax(axis=1)).astype(np.float32).mean()


