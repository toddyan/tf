import tensorflow as tf
from seq2seq_model import NMTModel

DATA_ROOT = "/Users/yxd/Downloads/en-zh/"
src_path = DATA_ROOT + "train.code.en"
trg_path = DATA_ROOT + "train.code.zh"
checkpoint_path = DATA_ROOT + "seq2seq.ckpt"

def main():
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()
    # "this is a test"
    test_sentence = [19, 13, 9, 709, 4, 2]
    # "who are you?"
    test_sentence = [83, 26, 14, 33, 2]
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