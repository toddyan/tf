##### -*- coding:utf-8 -*-
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


def get_tuned_variables(exclude_scopes):
    #debug
    print("--- slim.get_model_variables ---")
    for var in slim.get_model_variables():
        print(var.op.name)
    variables_to_restore = []
    for var in slim.get_model_variables():
        exclude = False
        for exclusion in exclude_scopes:
            if var.op.name.startswith(exclusion):
                exclude = True
        if not exclude:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainable_variables(trainable_scopes):
    variables_to_train = []
    for scope in trainable_scopes:
        variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main(argv=None):
    input_data = '/Users/yxd/Downloads/flower_photos.npy'
    save_model_path = '/tmp/flower_model'
    # download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    ckpt_file = '/Users/yxd/Downloads/inception_v3.ckpt'

    learning_rate = 0.00001
    epochs = 10
    batch_size = 32
    classes = 5

    exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    trainable_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    #debug
    for k,v in inception_v3.inception_v3_arg_scope().items():
        print(k)
        for k2,v2 in v.items():
            print("  " + k2)
            if isinstance(v2,dict):
                for k3,v3 in v2.items():
                    print("  " + k3 + ": " + str(v3))
            else:
                print(str(v2))
    npy_data = np.load(input_data)
    training_iamges = np.array(npy_data[0])
    training_labels = np.array(npy_data[1])
    validation_images = np.array(npy_data[2])
    validation_labels = np.array(npy_data[3])
    testing_images = np.array(npy_data[4])
    testing_labels = np.array(npy_data[5])
    n_training = len(training_iamges)
    print(np.array(training_iamges).shape)
    print(validation_images.shape)
    print(testing_images.shape)
    print(training_labels.shape)
    print(validation_labels.shape)
    print(testing_labels.shape)

    images = tf.placeholder(tf.float32, (None, 299, 299, 3), name="input-images")
    labels = tf.placeholder(tf.int64, (None,), name="input-labels")


    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=classes)
    trainable_variables = get_trainable_variables(trainable_scopes)
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, classes), logits, weights=1.0)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
        acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    load_fn = slim.assign_from_checkpoint_fn(ckpt_file,get_tuned_variables(exclude_scopes),ignore_missing_vars=True)

    saver = tf.train.Saver()

    with tf.Session() as s:
        tf.global_variables_initializer().run()
        print("loading tuned variables from %s" % ckpt_file)
        load_fn(s)
        global_step = 0
        for epoch in range(epochs):
            start = 0
            while start < training_iamges.shape[0]:
                end = min(start+batch_size, training_iamges.shape[0])
                s.run(optimizer, feed_dict={images:training_iamges[start:end], labels:training_labels[start:end]})
                global_step += 1
                start = end
                print("step " + str(global_step))
                if global_step%5==0:
                    saver.save(s, save_model_path, global_step=global_step)
                    validation_acc = s.run(acc, feed_dict={images:validation_images,labels:validation_labels})
                    print("step %d:validation acc %f" % (global_step, validation_acc))
            test_acc = s.run(acc, feed_dict={images:testing_images, labels:testing_labels})
            print("epoch %d:testing acc %f" % (epoch, test_acc))
if __name__ == "__main__":
    tf.app.run()