import tensorflow as tf
x  = tf.constant([[0.7,0.9]])
w1 = tf.Variable(tf.random_normal((2,3),stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))
a1 = tf.matmul(x,  w1)
y  = tf.matmul(a1, w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(y))