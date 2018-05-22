import tensorflow as tf
input_file = ['/tmp/text1','/tmp/text2']
dataset = tf.data.TextLineDataset(input_file)
iter = dataset.make_one_shot_iterator()
line = iter.get_next()
with tf.Session() as s:
    tf.global_variables_initializer().run()
    while True:
        try:
            txt = s.run(line)
            print(txt)
        except tf.errors.OutOfRangeError:
            break