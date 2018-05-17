import tensorflow as tf
reader = tf.TFRecordReader()
save_path="/tmp/mnist_tfrecord"
filename_queue = tf.train.string_input_producer([save_path])


_, serialized_img = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_img,
    features={
        'pixels': tf.FixedLenFeature(shape=[],dtype=tf.int64),
        'label': tf.FixedLenFeature(shape=[],dtype=tf.int64),
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
)
image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(image,(28,28))
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)
#image_batch, label_batch = tf.train.batch([image,label],batch_size=4)
image_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=4,capacity=512,min_after_dequeue=256)
with tf.Session() as s:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s,coord=coord)
    tf.global_variables_initializer().run()
    x,y = s.run([image_batch,label_batch])
    print x.shape
    print y



'''
_, serialized_img = reader.read_up_to(queue=filename_queue, num_records=2)
features = tf.parse_example(
    serialized_img,
    features={
        'pixels': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
)
image = tf.decode_raw(features['image'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)
with tf.Session() as s:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s,coord=coord)
    for i in range(2):
        # print(s.run([image,label,pixels]))
        image_data,label_data,pixels_data = s.run([image,label,pixels])
        print image_data.shape #(2, 784)
        print label_data #[5 0]
'''
