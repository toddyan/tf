import tensorflow as tf
files = tf.train.match_filenames_once("/tmp/examples-*")
file_queue = tf.train.string_input_producer(string_tensor=files, shuffle=False, num_epochs=2)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
features = tf.parse_single_example(serialized_example,features={
    "i":tf.FixedLenFeature([],tf.int64),
    "j":tf.FixedLenFeature([],tf.int64)
})

with tf.Session() as s:
    tf.local_variables_initializer().run()
    print(files.eval())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s,coord=coord)
    for i in range(6):
        print(s.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)