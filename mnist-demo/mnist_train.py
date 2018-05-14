# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import mnist_infer
import os
from keras.datasets import mnist
batch_size           = 64
learning_rate_base   = 0.8
learning_rate_decay  = 0.99
regularization_rate  = 0.0001
training_epochs      = 10000
moving_average_decay = 0.99
model_savepath       = "/tmp/tf/mnist"
model_name           = "mnist.ckpt"

def train(x_train, y_train):
    x = tf.placeholder(tf.float32, shape=(None,mnist_infer.N[0]),  name="input-x")
    y = tf.placeholder(tf.float32, shape=(None,mnist_infer.N[-1]), name="input-y")
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    z = mnist_infer.infer(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    ema_updater = ema.apply(tf.trainable_variables())
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    )
    loss = cross_entropy + tf.add_n(tf.get_collection("loss"))
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        x_train.shape[0]/batch_size,
        learning_rate_decay
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([ema_updater,optimizer]):
        trainer = tf.no_op(name="trainer")
    saver = tf.train.Saver()
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        for epoch in range(training_epochs):
            start = 0
            while start < x_train.shape[0]:
                end = min(x_train.shape[0], start + batch_size)
                xs  = x_train[start:end]
                ys  = y_train[start:end]
                _, loss_value, step = s.run([trainer, loss, global_step], feed_dict={x:xs, y:ys})
                start = end
            z_train = s.run(z,feed_dict={x:x_train})
            acc     = (z_train.argmax(axis=1)==y_train.argmax(axis=1)).mean()
            print("epoch[%d] loss=%f acc=%f" % (epoch, loss_value, acc))
            saver.save(s, os.path.join(model_savepath, model_name), global_step=epoch)

def main(argv=None):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],-1)/255.0
    x_test  = x_test.reshape(x_test.shape[0],-1)/255.0
    y_train = (y_train.reshape(y_train.shape[0],-1)==np.arange(10)).astype(np.float32)
    y_test  = (y_test.reshape(y_test.shape[0],-1)==np.arange(10)).astype(np.float32)
    training_size = int(0.9 * x_train.shape[0])
    x_valid = x_train[training_size:]
    y_valid = y_train[training_size:]
    x_train = x_train[0:training_size]
    y_train = y_train[0:training_size]

    train(x_train,y_train)

if __name__ == '__main__':
    tf.app.run()