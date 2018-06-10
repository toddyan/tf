import tensorflow as tf
MAX_LEN = 50
SOS_ID = 1

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda line: tf.string_split([line]).values)
    dataset = dataset.map(lambda str: tf.string_to_number(str, tf.int32))
    dataset = dataset.map(lambda x:(x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path) # shape: [examples, (array[len], 1)]
    trg_data = MakeDataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data,trg_data)) # shape: [example, ((array[len1],1),(array[len2],1))]
    def length_fiter(src_tuple, trg_tuple):
        ((src_input, src_len),(trg_label,trg_len)) = (src_tuple, trg_tuple)
        return tf.logical_and(
            tf.logical_and(
                tf.greater(src_len, 1),
                tf.less_equal(src_len, MAX_LEN)
            ),
            tf.logical_and(
                tf.greater(trg_len, 1),
                tf.less_equal(trg_len, MAX_LEN)
            )
        )
    dataset = dataset.filter(length_fiter)
    def dup_target_input(src_tuple, trg_tuple):
        ((src_input, src_len),(trg_label,trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len),(trg_input, trg_label, trg_len))
    dataset = dataset.map(dup_target_input) # shape: [example, ((array[len1],1),(array[len2],array[len2],1))]
    dataset = dataset.shuffle(10000)
    padding_shape = ((tf.TensorShape([None]), tf.TensorShape([])),
                     (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))
    # [example/batch, (([batch,len1],[batch]),([batch,len2],[batch,len2],[batch]))]
    dataset = dataset.padded_batch(batch_size, padding_shape)
    return dataset


if __name__ == "__main__":
    a0 = tf.constant("1 2 3",dtype=tf.string)
    a1 = tf.string_split([a0]).values
    a2 = tf.string_to_number(a1,tf.int32)
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        print(s.run([a0, a1, a2]))
        print("")

    DATA_ROOT = "/Users/yxd/Downloads/en-zh/"
    src_path = DATA_ROOT + "train.code.en"
    trg_path = DATA_ROOT + "train.code.zh"
    batch_size = 3

    ds = MakeSrcTrgDataset(src_path, trg_path, batch_size)
    iter = ds.make_one_shot_iterator()
    rec = iter.get_next()
    with tf.Session() as s:
        while True:
            print(s.run(rec))
            print("")
            print(s.run(rec))
            exit(0)

