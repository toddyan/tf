import numpy as np
import tensorflow as tf
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
TIMESTEPS = 10
NUM_LAYERS = 2
HIDDEN_SIZE = 30
TRAINING_STEPS = 10000
TESTING_EXAMPLES = 1000
TRAINING_EXAMPLES = 10000
SAMPLE_GAP = 0.01
BATCH_SIZE=32
def gen_data(seq):
    X = []
    Y = []
    for i in range(len(seq) - 10):
        X.append([seq[i:i+TIMESTEPS]])
        Y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
def model(X, y, training):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
    )
    #help(tf.nn.dynamic_rnn)
    outputs,c_h = tf.nn.dynamic_rnn(cell=cell, inputs=X, sequence_length=None, dtype=tf.float32)
    # outputs' dimesion:[ batch_size, time_step, last_hidden_size]
    output = outputs[:,-1,:]
    #help(tf.contrib.layers.fully_connected)
    predictions = tf.contrib.layers.fully_connected(inputs=output, num_outputs=1, activation_fn=None)
    if not training:
        return predictions,None,None
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    #help(tf.contrib.layers.optimize_loss)
    optimizer = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer="Adagrad",
        learning_rate=0.1
    )
    return predictions, loss, optimizer

def train(s, train_X, train_Y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model",reuse=False):
        predictions, loss, optimizer = model(X, y, True)
        s.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            _, lo = s.run([optimizer, loss])
            if i % 100 == 0:
                print("train step " + str(i) + ",loss:" + str(lo))

def eval(s, test_X, test_Y):
    ds = tf.data.Dataset.from_tensor_slices((test_X,test_Y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = model(X, [0.0], False)
        predictions = []
        labels = []
        for i in range(TESTING_EXAMPLES):
            pr, la = s.run([prediction, y])
            predictions.append(pr)
            labels.append(la)
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
        print("mse:",rmse)
        plt.figure()
        plt.plot(predictions, label="predictions")
        plt.plot(labels, label="sin")
        plt.legend()
        plt.show()

test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
seq = np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)
train_X, train_Y = gen_data(np.sin(seq)**2*2+np.cos(seq))
np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)
test_X, test_Y = gen_data(np.sin(seq)**2*2+np.cos(seq))

with tf.Session() as s:
    train(s, train_X, train_Y)
    eval(s, test_X, test_Y)