import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import mnist

def LeNet5(input):
    net = tf.reshape(tensor=input, shape=[-1, 28, 28, 1])
    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[5,5], padding="SAME", scope="layer1")
    net = slim.max_pool2d(inputs=net, kernel_size=2, stride=2, scope="layer2")
    net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[5,5], padding="SAME", scope="layer3")
    net = slim.max_pool2d(inputs=net, kernel_size=2, stride=2, scope="layer4")
    net = slim.flatten(inputs=net, scope="flatten")
    net = slim.fully_connected(inputs=net, num_outputs=500, scope="layer5")
    logits = slim.fully_connected(inputs=net, num_outputs=10, scope="output")
    return logits


def run_epoch(X,Y,learnint_rate, batch_size):
    input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="input")
    output = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="output")
    logits = LeNet5(input)
    cross_enytopy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=output,
        logits=logits
    ))
    optimizer = tf.train.AdamOptimizer(learning_rate=learnint_rate).minimize(cross_enytopy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, axis=1),
        tf.argmax(output, axis=1)
    ), tf.float32))
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        start = 0
        while start < X.shape[0]:
            end = start+batch_size
            acc, loss, _ = s.run([accuracy, cross_enytopy, optimizer], feed_dict={input:X[start:end],output:Y[start:end]})
            #if(start/batch_size==10):
            print(acc, loss)
            start = end

def main():
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    X_train = X_train.reshape([X_train.shape[0],-1])/255.0
    X_test = X_test.reshape([X_test.shape[0],-1])/255.0
    Y_train = (Y_train[:,np.newaxis] == np.arange(10)).astype(np.float32)
    Y_test = (Y_test[:, np.newaxis] == np.arange(10)).astype(np.float32)
    batch_size = 64
    learnint_rate = 0.001
    for epoch in range(10):
        run_epoch(X_train, Y_train, learnint_rate, batch_size)

if __name__ == "__main__":
    main()