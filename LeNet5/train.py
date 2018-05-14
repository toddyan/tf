import tensorflow as tf
import numpy as np
import os
import infer
from mnist_conf import Conf

def train(c):
    x = tf.placeholder(tf.float32, shape=(None,) + c.image_shape,  name="input-x")
    y = tf.placeholder(tf.float32, shape=(None,c.classes), name="input-y")
    regularizer = tf.contrib.layers.l2_regularizer(c.regularization_rate)
    z = infer.infer(c, x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
    ema = tf.train.ExponentialMovingAverage(c.moving_average_decay,global_step)
    ema_updater = ema.apply(tf.trainable_variables())
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    )
    loss = cross_entropy + tf.add_n(tf.get_collection("loss"))
    learning_rate = tf.train.exponential_decay(
        c.learning_rate_base,
        global_step,
        c.x_train.shape[0]/c.batch_size,
        c.learning_rate_decay
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([ema_updater,optimizer]):
        trainer = tf.no_op(name="trainer")
    saver = tf.train.Saver()
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        for epoch in range(c.training_epochs):
            start = 0
            while start < c.x_train.shape[0]:
                end = min(c.x_train.shape[0], start + c.batch_size)
                xs  = c.x_train[start:end]
                ys  = c.y_train[start:end]
                _, loss_value, step = s.run([trainer, loss, global_step], feed_dict={x:xs, y:ys})
                start = end
                if(step%10==0):
                    z_train = s.run(z,feed_dict={x:c.x_train})
                    acc     = (z_train.argmax(axis=1)==c.y_train.argmax(axis=1)).mean()
                    print("epoch[%d] loss=%f acc=%f" % (epoch, loss_value, acc))
                    saver.save(s, os.path.join(c.model_savepath, c.model_name), global_step=epoch)

def main(argv=None):
    train(Conf())

if __name__ == '__main__':
    tf.app.run()