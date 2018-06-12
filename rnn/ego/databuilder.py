# -*- coding: utf8 -*-
import tensorflow as tf
import codecs
import random

def gen_file(train_path, valid_path):
    fib = [1, 1]
    for _ in range(40):
        fib.append(fib[-2]+fib[-1])
    fib = [str(e) for e in fib]
    print(fib)
    with codecs.open(train_path,'w', encoding='utf-8') as f:
        for i in range(1, 30, 1):
            if i % 8 < 7:
                len = random.randint(5,10)
                f.write(' '.join(fib[i:i+len]) + "\n")
    with codecs.open(valid_path, 'w', encoding='utf-8') as f:
        for i in range(1, 30, 1):
            if i % 8 >= 7:
                len = random.randint(5, 10)
                f.write(' '.join(fib[i:i+len]) + "\n")

def make_data(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda line: tf.string_split([line]).values)
    dataset = dataset.map(lambda arr: (
        tf.string_to_number(arr,out_type=tf.int64),
        tf.size(arr)
    ))
    dataset = dataset.shuffle(100)
    dataset = dataset.padded_batch(5,padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])))
    return dataset


if __name__ == "__main__":
    ROOT="E:/Download/tf/"
    train_path = ROOT + "train"
    valid_path = ROOT + "valid"

    gen_file(train_path, valid_path)
    ds = make_data(train_path)
    iterator = ds.make_one_shot_iterator()
    data = iterator.get_next()
    a = tf.constant([[1,2],[3,4],[5,6]])
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        print(s.run(tf.shape(a)))
        print(s.run(tf.size(a)))
        while True:
            try:
                print(s.run(data))
            except tf.errors.OutOfRangeError:
                break
