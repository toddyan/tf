import sys
sys.path.append("..")
import globalconf
import tensorflow as tf
# a = tf.constant([1.0,2.0,3.0], name="a")
# b = tf.Variable(tf.random_normal([3]), name="b")
with tf.variable_scope("input1"):
    a = tf.constant([1.0,2.0,3.0], name="a")
with tf.variable_scope("input2"):
    b = tf.Variable(tf.random_normal([3]),name="b")
c = tf.add_n([a,b], name="c")
writer = tf.summary.FileWriter(globalconf.get_root()+"tensorboard/demo1", tf.get_default_graph())
writer.close()
