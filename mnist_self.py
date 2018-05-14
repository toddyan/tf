import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train_rs = x_train.reshape(x_train.shape[0], -1)/255.0
x_test_rs  = x_test.reshape(x_test.shape[0], -1)/255.0
y_train_rs = (y_train.reshape(y_train.shape[0], 1) == np.arange(10)).astype(np.float32)
y_test_rs  = (y_test.reshape(y_test.shape[0], 1) == np.arange(10)).astype(np.float32)


def getWeight(shape,lamb):
    w = tf.Variable(tf.clip_by_value(tf.random_normal(shape,stddev=0.1),clip_value_min=-0.1,clip_value_max=0.1),dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamb)(w))
    return w

x = tf.placeholder(dtype=np.float32, shape=(None,784), name="input-x")
y = tf.placeholder(dtype=np.float32, shape=(None,10), name="input-y")
batch_size = 64
epochs = 100

N = [784,800,800,800,50,20,10]
for n in range(1,len(N)):
    w = getWeight((N[n-1],N[n]),0.0001)
    if n == 1:
        z = tf.matmul(x, w) + tf.Variable(tf.fill((N[n],),0.1))
    else:
        z = tf.matmul(a, w) + tf.Variable(tf.fill((N[n],),0.1))
    if n + 1 != len(N):
        a = tf.nn.relu(z)
    else:
        crossEnt = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z,labels=y))
tf.add_to_collection("losses",crossEnt)
totalLoss = tf.add_n(tf.get_collection("losses"))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(totalLoss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        start = 0
        while start < x_train_rs.shape[0]:
            end = min(x_train_rs.shape[0], start + batch_size)
            sess.run(optimizer,feed_dict={x:x_train_rs[start:end],y:y_train_rs[start:end]})
            start = end
        print(sess.run(crossEnt,feed_dict={x:x_train_rs,y:y_train_rs}))
        z_pred = sess.run(z, feed_dict={x: x_train_rs})
        print(z_pred.argmax(axis=1)==y_train_rs.argmax(axis=1)).astype(np.float32).sum()/x_train_rs.shape[0]
        z_pred = sess.run(z, feed_dict={x: x_test_rs})
        print(z_pred.argmax(axis=1)==y_test_rs.argmax(axis=1)).astype(np.float32).sum()/x_test_rs.shape[0]
