import tensorflow as tf
import tensorflow.contrib.slim as slim
with tf.variable_scope("myVar"):
    v1 = tf.get_variable("v1",initializer=tf.zeros_initializer(dtype=tf.float32),shape=(2,))
    v2 = tf.Variable(initial_value=tf.ones(shape=(1,)),name="v2")
    sum = tf.add(v1,v2,name="sum")
    print(v1.name) # myVar/v1:0
    print(v2.name) # myVar/v2:0
    print(sum.name) # myVar/sum:0

print("----")
with tf.variable_scope("myVar",reuse=tf.AUTO_REUSE):
    v1 = tf.get_variable("v1",initializer=tf.zeros_initializer(dtype=tf.float32),shape=(2,))
    v2 = tf.Variable(initial_value=tf.ones(shape=(1,)),name="v2")
    sum = tf.add(v1,v2,name="sum")
    print(v1.name) # myVar/v1:0
    print(v2.name) # myVar_1/v2:0
    print(sum.name) # myVar_1/sum:0

print("----")
for var in slim.get_model_variables():
    print(var.name)

print("----")
for var in tf.trainable_variables():
    print(var.name)
# see https://blog.csdn.net/Jerr__y/article/details/70809528