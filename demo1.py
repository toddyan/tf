import numpy as np
import tensorflow as tf
m = 100000
batch_size = 64
epochs = 100
learning_rate = 0.001
x_data = np.random.rand(m,2)
y_data = (x_data.sum(axis=1, keepdims=1)>1).astype(np.float32)


w1 = tf.Variable(tf.random_normal((2,4),stddev=1,seed=1))
b1 = tf.Variable(tf.zeros((4,)))
w2 = tf.Variable(tf.random_normal((4,1),stddev=1,seed=1))
b2 = tf.Variable(tf.zeros(1,))

x = tf.placeholder(tf.float32,shape=(None,2),name="input-x")
y = tf.placeholder(tf.float32,shape=(None,1),name="input-y")

a  = tf.nn.relu(tf.matmul(x, w1) + b1)
y_ = tf.nn.relu(tf.matmul(a, w2) + b2)

loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_, clip_value_min=1e-10, clip_value_max=1.0))
                       + (1-y)*tf.log(tf.clip_by_value(1-y_, clip_value_min=1e-10, clip_value_max=1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        start = 0
        while start < m:
            end = min(m,start+batch_size)
            sess.run(optimizer,feed_dict={x:x_data[start:end], y:y_data[start:end]})
            start = end
        y_pred = sess.run(y_,feed_dict={x:x_data})
        precision = ((y_pred>0.5)==(y_data>0.5)).astype(np.float32).sum()/m
        print("epoch ",epoch,sess.run(loss,feed_dict={x:x_data,y:y_data}),precision)
        #print("w1:",sess.run(w1))
        #print("w2:",sess.run(w2))