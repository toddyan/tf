import numpy as np
import tensorflow as tf
import codecs

HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LSTM_KEEP_PROB = 0.9
INPUT_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_SOFTMAX = True


class PTBModel(object):
    def __init__(self, is_training, batch_size, timestep):
        self.batch_size = batch_size
        self.timestep = timestep

        self.input = tf.placeholder(tf.int32, shape=(batch_size, timestep))
        self.target = tf.placeholder(tf.int32, shape=(batch_size, timestep))
        keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        cells = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=keep_prob
            ) for _ in range(NUM_LAYERS)
        ])
        # use to inital first batch for each epoch
        self.initial_state = cells.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding",shape=(VOCAB_SIZE, HIDDEN_SIZE),dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.input) #[batch_size, timestep, hidden_size]
        if is_training:
            inputs = tf.nn.dropout(inputs, INPUT_KEEP_PROB)
        # collect all output of all timestep
        outputs = [] #timestep, batch, hidden_size
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for step in range(timestep):
                if step>0:tf.get_variable_scope().reuse_variables()
                cell_output, state = cells(inputs=inputs[:,step,:],state=state)
                outputs.append(cell_output)
        '''
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        # (t1.d0+t2.d0) * d1
        tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # d0 * (t1.d1+t2.d1)
        tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        '''
        output = tf.reshape(
            # [timestep, batch, hidden_size] -> [batch, timestep*hidden]
            # or [[:,0,:].reshape(-1), [:,1,:].reshape(-1) ...]
            tf.concat(outputs,1),
            # (batch*timestep, hidden_size)
            shape=(-1,HIDDEN_SIZE)
        )
        if SHARE_EMB_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", shape=(HIDDEN_SIZE, VOCAB_SIZE))
        bias = tf.get_variable("bias", shape=(VOCAB_SIZE,))
        logits = tf.matmul(output, weight) + bias #[batch_size*timestep, vocab_size]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(tensor=self.target, shape=(-1,)),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(
                tf.reshape(tensor=self.target,shape=(-1,)),
                tf.argmax(logits,axis=1,output_type=tf.int32)
            ), dtype=tf.float32)
        )
        if not is_training : return
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            t_list = tf.gradients(self.cost, trainable_variables),
            clip_norm=MAX_GRAD_NORM
        ) #[#trainable_variables]
        self.grads = grads #debug
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
        self.optimizer = optimizer.apply_gradients(zip(grads, trainable_variables))
def run_epoch(session, model, batches, optimizer, logging, step):
    total_cost = 0.0
    iters = 0
    state = session.run(model.initial_state)
    accs = []
    for x, y in batches:
        cost, state, _ = session.run(
            [model.cost, model.final_state, optimizer],
            feed_dict={
                model.input: x, # shape=(batch_size, timestep)
                model.target: y, # shape=(batch_size, timestep)
                model.initial_state: state
            }
        )
        acc = session.run(model.acc,feed_dict={
                model.input: x, # shape=(batch_size, timestep)
                model.target: y, # shape=(batch_size, timestep)
                model.initial_state: state
            })
        accs.append(acc)
        total_cost += cost
        iters += model.timestep
        #print("step %d, cost=%.3f, acc=%.3f" % (step, cost/model.timestep, acc))
        if logging and step % 50 == 0:
            print("step %d, perplexity=%.3f, acc=%.3f" % (step, np.exp(total_cost/iters),acc))
            #print("step %d, perplexity=%.3f" % (step, total_cost / iters))
        step += 1
    #return step, total_cost/iters
    return step, np.exp(total_cost/iters), np.mean(np.array(accs))
def read_data(file):
    with codecs.open(file, 'r', 'utf-8') as f:
        total = ' '.join([line.strip() for line in f.readlines()])
    return [int(w) for w in total.split()]
def make_batch(id_list, batch_size, timestep):
    num_batches = (len(id_list)-1) // (batch_size*timestep)
    data = np.reshape(
        np.array(id_list[:num_batches*batch_size*timestep]),
        [batch_size, num_batches*timestep]
    )
    #split [batch_size, num_batches*timestep] into num_batches * [batch_size, timestep]
    data_batches = np.split(data, num_batches, axis=1)
    label = np.reshape(
        np.array(id_list[1:num_batches*batch_size*timestep+1]),
        [batch_size, num_batches*timestep]
    )
    label_batches = np.split(label, num_batches, axis=1)
    return list(zip(data_batches,label_batches))
def main():
    train_data = "/Users/yxd/Downloads/simple-examples/data/ptb.train.code"
    valid_data = "/Users/yxd/Downloads/simple-examples/data/ptb.valid.code"
    test_data = "/Users/yxd/Downloads/simple-examples/data/ptb.test.code"

    train_data = "E:/Download/simple-examples/data/ptb.train.code"
    valid_data = "E:/Download/simple-examples/data/ptb.valid.code"
    test_data = "E:/Download/simple-examples/data/ptb.test.code"

    BATCH_SIZE = 30
    TIMESTEP = 35

    EPOCHS = 5


    initializer = tf.random_uniform_initializer(minval=0.05, maxval=0.15)
    with tf.variable_scope("language_model",reuse=None, initializer=initializer):
        train_model = PTBModel(True, BATCH_SIZE, TIMESTEP)
    with tf.variable_scope("language_model",reuse=True, initializer=initializer):
        eval_model = PTBModel(False, 1, 1)

    with tf.Session() as s:
        tf.global_variables_initializer().run()
        train_batches = make_batch(read_data(train_data),BATCH_SIZE,TIMESTEP)
        valid_batches = make_batch(read_data(valid_data),1,1)
        test_batches  = make_batch(read_data(test_data),1,1)
        step = 0
        for i in range(EPOCHS):
            print("Iter %d" % (i+1))
            step, pplx, acc = run_epoch(s,train_model,train_batches,train_model.optimizer,True,step)
            print("train perplexity:%.3f, acc:%.3f" % (pplx,acc))
            _, pplx, acc = run_epoch(s,eval_model,valid_batches,tf.no_op(),False,0)
            print("valid perplexity:%.3f, acc:%.3f" % (pplx, acc))
        _, pplx, acc = run_epoch(s, eval_model, test_batches, tf.no_op(), False, 0)
        print("test perplexity:%.3f, acc:%.3f" % (pplx, acc))

if __name__ == "__main__":
    main()
