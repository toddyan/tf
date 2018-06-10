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
MAX_DEC_LEN = 100
SOS_ID = 1
EOS_ID = 2
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
        for v in trainable_variables:  # TODO
            print(v)
        grads = tf.gradients(ys=cost/tf.to_float(batch_size),xs=trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        updater = optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables))
        return cost_per_token, updater

    def infer(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)],dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input],dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_emb, src_input)
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                cell=self.enc_cell,
                inputs=src_emb,
                sequence_length=src_size,
                dtype=tf.float32
            )
        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            def loop_cond(state, dec_ids, step):
                #return tf.convert_to_tensor(False,dtype=tf.bool)
                return tf.reduce_all(
                    tf.logical_and(
                        tf.not_equal(dec_ids.read(step), EOS_ID),
                        tf.less(step, MAX_DEC_LEN-1)
                    )
                )

            def loop_body(prev_step_state, dec_ids, step):
                #enc_ids.write(0, SOS_ID)
                prev_step_input = [dec_ids.read(step)]
                prev_step_emb = tf.nn.embedding_lookup(self.trg_emb, prev_step_input)
                cur_step_ouputs, cur_step_state = self.dec_cell.call(
                    inputs=prev_step_emb,
                    state=prev_step_state
                )
                logits = tf.matmul(
                    tf.reshape(cur_step_ouputs, (-1, HIDDEN_SIZE)),
                    self.softmax_weight
                ) + self.softmax_bias
                max_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                dec_ids = dec_ids.write(step+1, max_id[0])
                return cur_step_state, dec_ids, step+1

            dec_ids = tf.TensorArray(
                dtype=tf.int32,
                size=0,
                dynamic_size=True,
                clear_after_read=False
            )
            dec_ids = dec_ids.write(0, SOS_ID)
            dec_state, dec_ids, step = tf.while_loop(
                cond=loop_cond,
                body=loop_body,
                loop_vars=(enc_state, dec_ids, 0)
            )
            return dec_ids.stack()


def main():
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()
    # "this is a test"
    test_sentence = [19, 13, 9, 709, 4, 2]
    # "who are you?"
    # test_sentence = [83, 26, 14, 33, 2]
    output_tensor = model.infer(test_sentence)
    with tf.Session() as s:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(DATA_ROOT)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(s, ckpt.model_checkpoint_path)
            output = s.run(output_tensor)
            print(output)
        else:
            print("Not CheckPoint file found.")


if __name__ == "__main__":
    main()