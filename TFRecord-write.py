import numpy as np
import tensorflow as tf
from keras.datasets import mnist
(x1,y1),(x2,y2) = mnist.load_data()
pixels = x1.shape[1] * x1.shape[2]
save_path="/tmp/mnist_tfrecord"
writer = tf.python_io.TFRecordWriter(save_path)
for (x,y) in zip(x1,y1):
    image = x.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels' :  tf.train.Feature(int64_list=tf.train.Int64List(value=[pixels])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }))
    writer.write(example.SerializeToString())
writer.close()