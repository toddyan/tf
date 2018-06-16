# -*- coding: utf8 -*-
import tensorflow as tf
from attention_model import AttentionNMTModel
from dataset_builder import MakeSrcTrgDataset


DATA_ROOT = "/Users/yxd/Downloads/en-zh/"
MODEL_ROOT = "/Users/yxd/Downloads/en-zh/att/"
src_path = DATA_ROOT + "train.code.en"
trg_path = DATA_ROOT + "train.code.zh"
checkpoint_path = MODEL_ROOT + "att.ckpt"



def main():
    with tf.variable_scope("att_nmt_model", reuse=None):
        model = AttentionNMTModel()
    # "this is a test."
    test_sentence = [19, 13, 9, 709, 4, 2]
    # "who are you?"
    test_sentence = [83, 26, 14, 33, 2]
    # "where are you?"
    test_sentence = [109, 26, 14, 33, 2]
    # "you are the apple in my eyes."
    test_sentence = [14, 26, 5, 3658, 12, 52, 607, 4, 2]
    output_tensor = model.infer(test_sentence)
    with tf.Session() as s:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_ROOT)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(s, ckpt.model_checkpoint_path)
            output = s.run(output_tensor)
            print(output)
        else:
            print("Not CheckPoint  file found.")


if __name__ == "__main__":
    main()