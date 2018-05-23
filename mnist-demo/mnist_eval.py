import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import time
import mnist_infer
import mnist_train

interval = 5
def evaluate(x_valid, y_valid):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, (None, mnist_infer.N[0]),  name="input-x")
        y = tf.placeholder(tf.float32, (None, mnist_infer.N[-1]), name="input-y")
        feed = {x:x_valid, y:y_valid}
        z = mnist_infer.infer(x, None)
        acc_count = tf.equal(tf.argmax(z, 1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(acc_count,tf.float32))
        name_map = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay).variables_to_restore()
        for k,v in name_map.items():
            print("\trestore: " + str(k) + " -> " + str(v))
        #saver = tf.train.Saver(name_map) #use moving average parameter
        saver = tf.train.Saver()
        while True:
            with tf.Session() as s:
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_savepath)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(s, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score = s.run(acc, feed_dict=feed)
                    print("epoch[%s] validation acc=%f" % (global_step, acc_score))
                else:
                    print("no checkpoint file found.")
                time.sleep(interval)


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

    evaluate(x_valid, y_valid)

if __name__ == '__main__':
    tf.app.run()