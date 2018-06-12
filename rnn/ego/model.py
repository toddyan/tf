# -*- coding: utf8 -*-
import tensorflow as tf
from databuilder import make_data

# ROOT = "E:/Download/tf/"
ROOT = "/Users/yxd/Downloads/tf/"
train_path = ROOT + "train"
valid_path = ROOT + "valid"
HIDDEN_SIZE = 32



class Model():
    def __init__(self):
        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(2)]
        )
    def get_updater(self, seq, size):
        outputs, states = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=seq,
            sequence_length=size,
            dtype=tf.float32
        )
        return outputs, states

def run_epoch(s, t1, t2, step):
    while True:
        try:
            print(s.run([t1,t2]))
        except tf.errors.OutOfRangeError:
            break

if __name__ == "__main__":
    dataset = make_data(train_path)
    iterator = dataset.make_one_shot_iterator()
    seq, size = iterator.get_next()
    seq = tf.expand_dims(input=seq, axis=2)
    with tf.variable_scope("model"):
        m = Model()
    # with tf.Session() as s:
    #     print("====")
    #     print(s.run([seq,size]))
    t1, t2 = m.get_updater(seq, size)
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        run_epoch(s, t1, t2, 0)
