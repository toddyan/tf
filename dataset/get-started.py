import tensorflow as tf
a = [1, 2, 3, 4, 5, 6, 7, 8]
dataset = tf.data.Dataset.from_tensor_slices(a)
iter = dataset.make_one_shot_iterator()
x = iter.get_next()
y = x * x
with tf.Session() as s:
    tf.global_variables_initializer().run()
    for _ in range(len(a)):
        print(s.run(y))