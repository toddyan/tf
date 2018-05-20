import tensorflow as tf
import numpy as np
import threading
import time


def MyThread(coord, work_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("stoping from thread %s" % work_id)
            coord.request_stop()
        else:
            print("Working[%s]" % work_id)
            time.sleep(1)
coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyThread,args=(coord,i)) for i in range(5)]
for t in threads: t.start()