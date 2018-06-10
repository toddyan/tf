# -*- coding: utf8 -*-
import tensorflow as tf
from dataset_builder import MakeSrcTrgDataset

DATA_ROOT = "/Users/yxd/Downloads/en-zh/"
src_path = DATA_ROOT + "train.code.en"
trg_path = DATA_ROOT + "train.code.zh"
checkpoint_path = DATA_ROOT + "seq2seq.ckpt"
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_SOFTMAX = True

class NMTModel(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
        ])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
        ])

        self.src_emb = tf.get_variable("src_emb",shape=[SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_emb = tf.get_variable("trg_emb",shape=[TRG_VOCAB_SIZE, HIDDEN_SIZE])
        if SHARE_EMB_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_emb)
        else:
            self.softmax_weight = tf.get_variable("softmax_weight",shape=[HIDDEN_SIZE,TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("softmax_bias",shape=[TRG_VOCAB_SIZE])
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        src_emb = tf.nn.embedding_lookup(params=self.src_emb, ids=src_input)
        trg_emb = tf.nn.embedding_lookup(params=self.trg_emb, ids=trg_input)
        src_emb = tf.nn.dropout(src_emb,keep_prob=KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb,keep_prob=KEEP_PROB)

        with tf.variable_scope("encoder"):
            src_outputs, src_state = tf.nn.dynamic_rnn(
                cell=self.enc_cell,
                inputs=src_emb,
                sequence_length=src_size,
                dtype=tf.float32
            )
        with tf.variable_scope("decoder"):
            #dec_outputs:[batch, timestep, hidden]
            dec_outputs, _ = tf.nn.dynamic_rnn(
                cell=self.dec_cell,
                inputs=trg_emb,
                sequence_length=trg_size,
                initial_state=src_state,
                dtype=tf.float32
            )
        ouput = tf.reshape(dec_outputs,shape=[-1, HIDDEN_SIZE])
        logits = tf.matmul(ouput,self.softmax_weight) + self.softmax_bias #[batch*timestep, trg_vocab]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label,[-1]),
            logits=logits
        )
        label_weights = tf.sequence_mask(lengths=trg_size,maxlen=tf.shape(trg_label)[1],dtype=tf.float32)
        label_weights = tf.reshape(label_weights, shape=[-1])
        cost = tf.reduce_sum(label_weights * loss)
        cost_per_token = cost/tf.reduce_sum(label_weights)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(ys=cost/tf.to_float(batch_size),xs=trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        updater = optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
        return cost_per_token, updater
def run_epoch(s, cost, updater, saver, step):
    while True:
        try:
            cost_value,_ = s.run([cost, updater])
            if step % 10 == 0:
                print("step %d, cost_per_token=%.3f" % (step, cost_value))
            if step % 100 == 0:
                saver.save(s, checkpoint_path, global_step = step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("nmt_model",reuse=None, initializer=initializer):
        train_model = NMTModel()
    data = MakeSrcTrgDataset(src_path, trg_path, BATCH_SIZE)
    iter = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iter.get_next()
    cost, updater = train_model.forward(src,src_size,trg_input,trg_label,trg_size)
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("Iter %d" % (i + 1))
            s.run(iter.initializer)
            step = run_epoch(s,cost,updater,saver,step)

if __name__ == "__main__":
    main()