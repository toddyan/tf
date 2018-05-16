import  tensorflow as tf
import tensorflow.contrib.slim as slim
with tf.name_scope("myname"):
    v1 = tf.get_variable("v1", shape=(1,), initializer=tf.zeros_initializer(dtype=tf.float32))
    v2 = tf.Variable(initial_value=tf.ones(shape=(2,)), name="v2")
    sum = tf.add(v1,v2,name="sum")
print(v1.name) # v1:0
print(v2.name) # myname/v2:0
print(sum.name) # myname/sum:0
print("----")
with tf.name_scope("myname2"):
    # ERROR, Variable v1 already exists, disallowed.
    # Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
    #v1 = tf.get_variable("v1",shape=(2,),initializer=tf.truncated_normal_initializer(stddev=0.1))
    print(v1.name)

print("----")
for var in slim.get_model_variables():
    print(var.name)

print("----")
for var in tf.trainable_variables():
    print(var.name)