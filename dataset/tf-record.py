import tensorflow as tf
import matplotlib.pyplot as plt
def parse(record):
    example = tf.parse_single_example(
        record,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
            "weight": tf.FixedLenFeature([], tf.int64),
            "height": tf.FixedLenFeature([], tf.int64),
            "channel": tf.FixedLenFeature([], tf.int64)
        }
    )
    image, label = example["image"], example["label"]
    height, weight, channel = example["height"], example["weight"], example["channel"]
    image_shape = tf.stack([height, weight, channel])
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, shape=image_shape)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    return image, label

method = 2
if method == 1:
    #input_files = ['/tmp/mnist_tfrecord']
    input_files = ['E:/tmp/mnist_tfrecord']
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parse)
    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        while True:
            try:
                img, lbl = s.run([image, label])
                plt.imshow(img)
                plt.title(lbl)
                plt.show()
            except tf.errors.OutOfRangeError:
                break
elif method == 2:
    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parse)
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()
    with tf.Session() as s:
        s.run([tf.global_variables_initializer(), iterator.initializer],
              #feed_dict={input_files: ['/tmp/mnist_tfrecord']})
              feed_dict={input_files: ['E:/tmp/mnist_tfrecord']})
        while True:
            try:
                img, lbl = s.run([image, label])
                plt.imshow(img)
                plt.title(lbl)
                plt.show()
            except tf.errors.OutOfRangeError:
                break

