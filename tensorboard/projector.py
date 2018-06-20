import sys
sys.path.append("..")
sys.path.append("../mnist-demo/")
import globalconf
import mnist_infer
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

REGULATIZATION_RATE = 0.0001
EMA_DECAY = 0.99

log_dir = globalconf.get_root() + "tensorboard/project"
sprite_image = globalconf.get_root() + "tensorboard/visual/sprite.png"
meta_path = globalconf.get_root() + "tensorboard/visual/meta.tsv"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

def train():
    with tf.variable_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="input-x")
        y = tf.placeholder(dtype=tf.int32, shape=[None], name="input-y")
    regularizer = tf.contrib.layers.l2_regularizer(REGULATIZATION_RATE)
    logits = mnist_infer.infer(x, regularizer)
    global_step = tf.get_variable("global_step",shape=0,initializer=tf.zeros_initializer)
    with tf.variable_scope("ema"):
        ema = tf.train.ExponentialMovingAverage(decay=EMA_DECAY, num_updates=global_step)
        ema_op = ema.apply(tf.trainable_variables())
train()
