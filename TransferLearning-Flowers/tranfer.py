# -*- coding:utf-8 -*-
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


def get_tuned_variables(exclude_scopes):
    variables_to_restore = []
    for var in slim.get_model_variables():
        exclude = False
        for exclusion in exclude_scopes:
            if var.op.name.startwith(exclusion):
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

def main():
    input_data = 'E:/Download/flower_photos.npy'
    save_model_path = 'E:/tmp/flower_model'
    # download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    ckpt_file = 'E:/Download/inception_v3/inception_v3.ckpt'

    learning_rate = 0.0001
    epochs = 10
    batch_size = 32
    classes = 5

    exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
    trainable_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    npy_data = np.load('E:/Download/flower_photos.npy')
    training_iamges = npy_data[0]
    training_labels = npy_data[1]
    validation_images = npy_data[2]
    validation_labels = npy_data[3]
    testing_images = npy_data[4]
    testing_labels = npy_data[5]
    n_training = len(training_iamges)
    print(training_iamges.shape)
    print(validation_images.shape)
    print(testing_images.shape)
    print(training_labels.shape)
    print(validation_labels.shape)
    print(testing_labels.shape)

    images = tf.placeholder(tf.float32, (None, 299, 299, 3), name="input-images")
    labels = tf.placeholder(tf.int64, (None,), name="input-labels")

    print(inception_v3.inception_v3_arg_scope())
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=classes)
    trainable_variables = get_trainable_variables(trainable_scopes)
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, classes), logits, weights=1.0)