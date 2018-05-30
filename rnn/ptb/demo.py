import numpy as np
import tensorflow as tf
train_data = "/Users/yxd/Downloads/simple-examples/data/ptb.train.code"
valid_data = "/Users/yxd/Downloads/simple-examples/data/ptb.valid.code"
test_data = "/Users/yxd/Downloads/simple-examples/data/ptb.test.code"

HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000
BATCH_SIZE = 20
TIMESTEP = 35

EPOCHS = 5
LSTM_KEEP_PROB = 0.9
INPUT_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_SOFTMAX = True

class PTBModel(object):
    def __init__(self, is_training, batch_size, timestep):
        self.batch_size = batch_size
        self.timestep = timestep

        self.input = tf.placeholder(tf.float32, shape=(batch_size, timestep))
        self.target = tf.placeholder(tf.float32, shape=(batch_size, timestep))
        keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        cells = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=keep_prob
            ) for _ in range(NUM_LAYERS)
        ])
        # use to inital first batch for each epoch
        self.initial_state = cells.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding",shape=(VOCAB_SIZE, HIDDEN_SIZE))
        inputs = tf.nn.embedding_lookup(embedding, self.input)
        if is_training:
            inputs = tf.nn.dropout(inputs, INPUT_KEEP_PROB)
        # collect all output of all timestep
        outputs = []