import tensorflow as tf
files = tf.train.match_filenames_once("/tmp/examples-*")
file_queue = tf.train.string_input_producer(string_tensor=files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
example = tf.parse_single_example(serialized_example,features={
    "i": tf.FixedLenFeature([], tf.int64),
    "j": tf.FixedLenFeature([], tf.int64)
})
x, y = example["i"], example["j"]
batch_size = 3
capacity = 1000 + batch_size * 3
x_batch, y_batch = tf.train.batch([x, y], batch_size=batch_size, capacity=capacity)
with tf.Session() as s:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s, coord=coord)
    for i in range(5):
        print(s.run([x_batch, y_batch]))
    coord.request_stop()
    coord.join(threads)