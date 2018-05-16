import tensorflow as tf

g1 = tf.Graph()
print(g1 == tf.get_default_graph()) #false
with g1.as_default():
    print(g1 == tf.get_default_graph()) #true
print(g1 == tf.get_default_graph()) #false
