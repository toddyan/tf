import tensorflow as tf
from seq2seq_model import NMTModel
from dataset_builder import MakeSrcTrgDataset
BATCH_SIZE = 100
NUM_EPOCH = 5

DATA_ROOT = "/Users/yxd/Downloads/en-zh/"
src_path = DATA_ROOT + "train.code.en"
trg_path = DATA_ROOT + "train.code.zh"
checkpoint_path = DATA_ROOT + "seq2seq.ckpt"

def run_epoch(s, cost, updater, saver, step):
    while True:
        try:
            cost_value,_ = s.run([cost, updater])
            if step % 10 == 0:
                print("step %d, cost_per_token=%.3f" % (step, cost_value))
            if step % 100 == 10:
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