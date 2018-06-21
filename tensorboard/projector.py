import sys
sys.path.append("..")
sys.path.append("../mnist-demo/")
import globalconf
import mnist_infer
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.contrib.tensorboard.plugins import projector

REGULATIZATION_RATE = 0.0001
EMA_DECAY = 0.99
learning_rate_base = 0.1
learning_rate_decay = 0.99
batch_size = 64
TRAINING_STEP = 10000
decay_steps = 1000


log_dir = globalconf.get_root() + "tensorboard/project/"
sprite_image = globalconf.get_root() + "tensorboard/visual/sprite.png"
meta_path = globalconf.get_root() + "tensorboard/visual/meta.tsv"


def train_definer():
    with tf.variable_scope("input"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="input-x")
        y = tf.placeholder(dtype=tf.int64, shape=[None], name="input-y")
    regularizer = tf.contrib.layers.l2_regularizer(REGULATIZATION_RATE)
    logits = mnist_infer.infer(x, regularizer)
    global_step = tf.get_variable(
        "global_step",
        shape=[],
        initializer=tf.zeros_initializer(),
        trainable=False
    )
    #global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope("ema"):
        ema = tf.train.ExponentialMovingAverage(decay=EMA_DECAY, num_updates=global_step)
        ema_op = ema.apply(tf.trainable_variables())
    with tf.variable_scope("loss"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits, axis=1), y
        ), tf.float32))
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
        )
        loss = cross_entropy + tf.add_n(tf.get_collection("loss"))
    with tf.variable_scope("optimizer"):
        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate_base,
            global_step = global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay,
            staircase=True
        )
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_op,ema_op]):
            agg = tf.no_op(name="agg")
    return x,y,logits,agg, accuracy, loss, global_step


def visualisation(test_logits):
    logits = tf.Variable(test_logits, name="LAST_LOGITS")
    summary_writer = tf.summary.FileWriter(log_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = logits.name
    embedding.metadata_path = meta_path
    embedding.sprite.image_path = sprite_image
    embedding.sprite.single_image_dim.extend([28,28])
    projector.visualize_embeddings(summary_writer,config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, log_dir+"model",TRAINING_STEP)
    summary_writer.close()


def main():
    x, y, logits, agg, accuracy, loss, global_step = train_definer()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        i = 0
        while i < TRAINING_STEP:
            start = 0
            while i < TRAINING_STEP and start < x_train.shape[0]:
                end = min(start+batch_size, x_train.shape[0])
                xs,ys = x_train[start:end], y_train[start:end]
                _,acc,lo,step = s.run([agg, accuracy, loss, global_step], feed_dict={x:xs,y:ys})
                start = end
                i += 1
                print(step,lo,acc)
        print("test acc:",s.run(accuracy, feed_dict={x:x_test, y:y_test}))
        test_logits = s.run(logits, feed_dict={x:x_test})
    visualisation(test_logits)

if __name__ == '__main__':
    main()

