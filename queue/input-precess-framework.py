import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("../ImagePreProcess")
import preprocess
sys.path.append("../LeNet5")
from comm import ConvLayer
import infer
class Conf:
    def __init__(self):
        # structure
        self.image_shape  = (28,28,3)
        self.classes      = 10
        self.conv_layers  = [ConvLayer((5,5,3,8), (1,1,1,1), (1,2,2,1), (1,2,2,1)),
                             ConvLayer((5,5,8,16),(1,1,1,1),(1,2,2,1),(1,2,2,1))]
        self.fc_layers    = [128, self.classes]

in_path = "/tmp/mnist_tfrecord"
target_height = 28
target_weight = 28
batch_size = 64
min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * batch_size
learning_rate = 0.00001

files = tf.train.match_filenames_once(in_path)
file_queue = tf.train.string_input_producer(string_tensor=files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
example = tf.parse_single_example(
    serialized_example,
    features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "weight": tf.FixedLenFeature([], tf.int64),
        "height": tf.FixedLenFeature([], tf.int64),
        "channel": tf.FixedLenFeature([], tf.int64)
    }
)
image,label = example["image"], example["label"]
height, weight, channel = example["height"], example["weight"], example["channel"]
image_shape = tf.stack([height, weight, channel])
image = tf.decode_raw(image,tf.uint8)
image = tf.reshape(image, shape=image_shape)
image = tf.image.convert_image_dtype(image, tf.float32)

image = tf.image.grayscale_to_rgb(image)
'''
image = preprocess.preprocess_for_train(
    image,
    target_height,
    target_weight,
    bbox=None,
    enable_hir_flip=False
)
'''
image = tf.image.resize_images(image, size=[target_height,target_weight])

image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    num_threads=2
)

logits = infer.infer(Conf(), image_batch, True, None)

cross_entropy = tf.reduce_mean(
    #tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=label_batch)
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batch)
)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), label_batch)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as s:
    s.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s, coord=coord)
    #ib, lb = s.run([image_batch, label_batch])
    for i in range(100000):
        s.run(optimizer)
        if i % 100 == 0 :
            acc = s.run(accuracy)
            print(acc)
    coord.request_stop()
    coord.join(threads)