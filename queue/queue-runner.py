import tensorflow as tf
queue = tf.FIFOQueue(100,"float")
pusher = queue.enqueue([tf.random_normal([3])])
qr = tf.train.QueueRunner(queue=queue,enqueue_ops=[pusher]*5)
tf.train.add_queue_runner(qr)
out = queue.dequeue()
with tf.Session() as s:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=s,coord=coord)
    for _ in range(10):
        print(s.run(out))
    coord.request_stop()
    coord.join(threads)