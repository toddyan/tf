import tensorflow as tf


v1 = tf.get_variable("v1",shape=(2,3),initializer=tf.constant_initializer(1,dtype=tf.float32))
v2 = tf.get_variable("v2",shape=(3,1),initializer=tf.constant_initializer(2,dtype=tf.float32))
product = tf.matmul(v1,v2,name="product")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(product))
    saver = tf.train.Saver()
    saver.save(sess,"/tmp/tf/model.ckpt")

'''
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"/tmp/tf/model.ckpt")
    print(sess.run(product))
'''

'''
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("/tmp/tf/model.ckpt.meta")
    saver.restore(sess,"/tmp/tf/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("product:0")))
'''