import numpy as np
import tensorflow as tf
from keras.datasets import mnist

def serialize(x,y):
    image = x.tostring()
    height = 28
    weight = 28
    channel = 1
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'weight': tf.train.Feature(int64_list=tf.train.Int64List(value=[weight])),
        'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
    }))
    return example.SerializeToString()

(x1,y1),(x2,y2) = mnist.load_data()
#save_path="/tmp/mnist_tfrecord"
save_path="E:/tmp/mnist_tfrecord"
writer = tf.python_io.TFRecordWriter(save_path)
for (x,y) in zip(x1,y1):
    writer.write(serialize(x,y))
writer.close()

#save_path="/tmp/mnist-tfrecord-test"
save_path="E:/tmp/mnist_tfrecord-test"
writer = tf.python_io.TFRecordWriter(save_path)
for (x,y) in zip(x2,y2):
    writer.write(serialize(x,y))
writer.close()