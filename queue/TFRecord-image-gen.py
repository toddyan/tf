import numpy as np
import tensorflow as tf
from keras.datasets import mnist
(x1,y1),(x2,y2) = mnist.load_data()
save_path="/tmp/mnist_tfrecord"
writer = tf.python_io.TFRecordWriter(save_path)
for (x,y) in zip(x1,y1):
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
    writer.write(example.SerializeToString())
writer.close()