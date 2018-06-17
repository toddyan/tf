import sys
sys.path.append("../../")
import globalconf
import numpy as np
import tensorflow as tf

from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)
root = globalconf.get_root()
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0],-1)).astype(np.float32)/255.0
X_test = X_test.reshape((X_test.shape[0],-1)).astype(np.float32)/255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)

image_shape = X_train.shape[1:]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, image_shape)

model_dir = root + "estimator/mnist"
feature_columns = [tf.feature_column.numeric_column("image",shape=image_shape)]
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500],
    feature_columns=feature_columns,
    model_dir=model_dir,
    n_classes=10,
    optimizer=tf.train.AdamOptimizer()
)

train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":X_train},
    y=Y_train,
    batch_size=64,
    num_epochs=1,
    shuffle=True
)

estimator.train(
    input_fn=train_fn,
    steps=10000
)

test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":X_test},
    y=Y_test,
    batch_size=64,
    num_epochs=1,
    shuffle=False
)

accuracy = estimator.evaluate(test_fn)['accuracy']

print(accuracy)