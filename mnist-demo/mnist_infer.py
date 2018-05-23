# -*- coding: utf-8 -*-
import tensorflow as tf

N = [784,500,10]

def get_weight_variable(shape, regularizer):
    w = tf.get_variable("w", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
    print(w)
    if regularizer != None:
        tf.add_to_collection("loss",regularizer(w))
    return w

def infer(x, regularizer):
    for i in range(1,len(N)):
        layer_name = "layer" + str(i)
        with tf.variable_scope(layer_name):
            w = get_weight_variable((N[i-1],N[i]),regularizer)
            b = tf.get_variable("b",shape=(N[i],), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            if i == 1:
                a = x
            z = tf.matmul(a, w) + b
            if i + 1 == len(N):
                return z
            a = tf.nn.relu(z)
