import sys
sys.path.append("../../")
import globalconf
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)

root = globalconf.get_root()

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train = X_train[:,:,:,np.newaxis]/255.0
X_test = X_test[:,:,:,np.newaxis]/255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
image_size = X_train.shape[1:]
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape, image_size)

def LeNet5(x, training):
    net = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.flatten(inputs=net)
    net = tf.layers.dense(inputs=net, units=500, activation=tf.nn.relu)
    net = tf.layers.dropout(inputs=net, rate=0.2, training=training)
    logits = tf.layers.dense(inputs=net, units=10)
    return logits

x = tf.placeholder(dtype=tf.float32, shape=(None,)+image_size)
y = tf.placeholder(dtype=tf.int32, shape=[None])
with tf.variable_scope("lenet5", reuse=None):
    logits = LeNet5(x, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), y),tf.float32))


with tf.variable_scope("lenet5",reuse=True):
    logits_valid = LeNet5(x, False)
    accuracy_valid = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_valid, axis=1, output_type=tf.int32), y),tf.float32))

with tf.Session() as s:
    tf.global_variables_initializer().run()
    for epoch in range(1,10):
        start = 0
        while start < X_train.shape[0]:
            end = min(start+64, X_train.shape[0])
            _, lo, acc = s.run([optimizer, loss, accuracy], feed_dict={x:X_train[start:end],y:Y_train[start:end]})
            start = end
            if(start/64%10==1):
                print(lo, acc)
            if(start/64%100==1):
                acc = s.run(accuracy_valid,feed_dict={x:X_test, y:Y_test})
                print(acc)
