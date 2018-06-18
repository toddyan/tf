import sys
sys.path.append("../..")
import globalconf
import numpy as np
import tensorflow as tf
import urllib


train_path = globalconf.get_root() + "estimator/iris/data/iris.train.csv"
test_path = globalconf.get_root() + "estimator/iris/data/iris.test.csv"

def gen_data():
    f_train = open(train_path, 'w')
    f_test = open(test_path, 'w')
    # http://archive.ics.uci.edu/ml/datasets/Iris
    label_dict = {}
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    for cnt,line in zip(range(65535),urllib.urlopen(url).read().split('\n')):
        tokens = line.split(',')
        if len(tokens) != 5: continue
        if not label_dict.has_key(tokens[4]):
            label_dict[tokens[4]] = str(len(label_dict))
        out_line = ','.join(tokens[:4]+[label_dict[tokens[4]]]) + '\n'
        if(cnt % 5 != 0):
            f_train.write(out_line)
        else:
            f_test.write(out_line)
    f_train.close()
    f_test.close()

def input_fn(path, shuffle, repeat):
    def csv_decoder(line):
        parsed_line = tf.decode_csv(records=line, record_defaults=[[0.],[0.],[0.],[0.],[0]])
        # return parsed_line
        return {"x":tf.stack(parsed_line[:-1])},parsed_line[-1]
    dataset = tf.data.TextLineDataset(path).map(csv_decoder)
    if(shuffle): dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(repeat).batch(8)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

###############################################
# gen_data()
# x,y = input_fn(train_path, True, 1)
# with tf.Session() as s:
#     tf.global_variables_initializer().run()
#     while True:
#         try:
#             print(s.run([x,y]))
#         except tf.errors.OutOfRangeError:
#             break
###############################################

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(
    hidden_units=[10,10],
    feature_columns=feature_columns,
    model_dir=globalconf.get_root()+"estimator/iris/model",
    n_classes=3
)
for epoch in range(100):
    classifier.train(input_fn=lambda: input_fn(train_path,True,1))
    result = classifier.evaluate(input_fn=lambda:input_fn(test_path,False,1))
    print(result)