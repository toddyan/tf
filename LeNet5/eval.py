import tensorflow as tf
import time
import infer
from cifar10_conf import Conf

interval = 15
def evaluate(c):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, (None, ) + c.image_shape,  name="input-x")
        y = tf.placeholder(tf.float32, (None, c.classes), name="input-y")
        feed = {x:c.x_valid, y:c.y_valid}
        z = infer.infer(c, x, False, None)
        acc_count = tf.equal(tf.argmax(z, 1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(acc_count,tf.float32))
        name_map = tf.train.ExponentialMovingAverage(c.moving_average_decay).variables_to_restore()
        for k,v in name_map.items():
            print ("\trestore: " + str(k) + " -> " + str(v))
        saver = tf.train.Saver(name_map) #use moving average parameter
        #saver = tf.train.Saver()
        while True:
            with tf.Session() as s:
                ckpt = tf.train.get_checkpoint_state(c.model_savepath)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(s, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score = s.run(acc, feed_dict=feed)
                    print("epoch[%s] validation acc=%f" % (global_step, acc_score))
                else:
                    print("no checkpoint file found.")
                time.sleep(interval)


def main(argv=None):
    evaluate(Conf())

if __name__ == '__main__':
    tf.app.run()