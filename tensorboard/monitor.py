# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import globalconf
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
batch_size           = 64
training_epochs      = 10
log_savepath         = globalconf.get_root() + "tensorboard/monitor/log"

def add_variable_to_monitor(var, name):
    with tf.variable_scope("summaries"):
        tf.summary.histogram(name=name, values=var)
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("mean/"+name, mean)
        tf.summary.scalar("stddev/"+name, stddev)

def dense(input_tensor, input_dim, output_dim, layer_name, act):
    with tf.variable_scope(layer_name):
        with tf.variable_scope("weights"):
            weights = tf.get_variable(
                "weights",
                shape=[input_dim,output_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
            )
            add_variable_to_monitor(weights, layer_name+"/weights")
        with tf.variable_scope("biases"):
            biases = tf.get_variable(
                "biases",
                shape=[output_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
            )
            add_variable_to_monitor(biases, layer_name+"/biases")
        with tf.variable_scope("logits"):
            logits = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name+"/logits",logits)
        activations = act(logits, name="activation")
        tf.summary.histogram(layer_name+"/activations",activations)
        return activations


def main(argv=None):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],-1)/255.0
    x_test  = x_test.reshape(x_test.shape[0],-1)/255.0
    y_train = (y_train.reshape(y_train.shape[0],-1)==np.arange(10)).astype(np.float32)
    y_test  = (y_test.reshape(y_test.shape[0],-1)==np.arange(10)).astype(np.float32)

    with tf.variable_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="input-x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="input-y")
    with tf.variable_scope("input_image"):
        images = tf.reshape(x,[-1,28,28,1])
        tf.summary.image("images",images,max_outputs=20)
    hidden = dense(x, 784, 500, "layer1", tf.nn.relu)
    logits = dense(hidden, 500, 10, "layer2", tf.identity)
    with tf.variable_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        tf.summary.scalar("cross_entropy",cross_entropy)
    with tf.variable_scope("train"):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
    with tf.variable_scope("eval"):
        with tf.variable_scope("correct_pred"):
            correct_pred = tf.equal(tf.argmax(logits,axis=1),tf.argmax(y,axis=1))
        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy",accuracy)
        merged = tf.summary.merge_all()
    with tf.Session() as s:
        summary_writer = tf.summary.FileWriter(log_savepath,s.graph)
        tf.global_variables_initializer().run()
        step = 0
        for epoch in range(training_epochs):
            start = 0
            while start < x_train.shape[0]:
                end = min(start + batch_size, x_train.shape[0])
                _, cro, acc, summaries = s.run(
                    [train_op, cross_entropy,accuracy,merged],
                    feed_dict={x:x_train[start:end],y:y_train[start:end]}
                )
                summary_writer.add_summary(summaries, step)
                step += 1
                start = end
            print(s.run(accuracy,feed_dict={x:x_test,y:y_test}))
if __name__ == "__main__":
    main()