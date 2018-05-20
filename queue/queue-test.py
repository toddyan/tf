import tensorflow as tf

q = tf.FIFOQueue(capacity=2,dtypes="int32")
#q = tf.RandomShuffleQueue(2,"int32")
init = q.enqueue_many(([0,10],))
x = q.dequeue()
y = x + 1
pusher = q.enqueue([y])

with tf.Session() as s:
    s.run(init)
    for _ in range(5):
        x_data, y_data,  _ = s.run([x, y, pusher])
        print(x_data)