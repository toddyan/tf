import sys
sys.path.append("../../")
import globalconf
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)

root = globalconf.get_root()

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train = X_train[:,:,:,np.newaxis].astype(np.float32)/255.0
X_test = X_test[:,:,:,np.newaxis].astype(np.float32)/255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
image_size = X_train.shape[1:]
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape, image_size)

def LeNet5(x, training):
    net = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.flatten(inputs=net)
    net = tf.layers.dense(inputs=net, units=500, activation=tf.nn.relu)
    net = tf.layers.dropout(inputs=net, rate=0.2, training=training)
    logits = tf.layers.dense(inputs=net, units=10)
    return logits

def model_fn(features, labels, mode, params):
    logits = LeNet5(features['image'], mode == tf.estimator.ModeKeys.TRAIN)
    pred = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"result":tf.argmax(logits, axis=1)}
        )
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())
    metrics = {"metrics":tf.metrics.accuracy(labels=labels, predictions=pred)}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=root + "estimator/cnn",
    params={"learning_rate":0.001}
)

train_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":X_train},
    y=Y_train,
    batch_size=64,
    num_epochs=10,
    shuffle=True
)
test_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":X_test},
    y=Y_test,
    batch_size=64,
    num_epochs=1,
    shuffle=False
)
pred_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":X_test[:100]},
    num_epochs=1,
    shuffle=False
)

estimator.train(input_fn=train_fn, steps=1000)
metrics = estimator.evaluate(input_fn=test_fn)['metrics']
print("acc:",metrics)
pred = estimator.predict(input_fn=pred_fn)
for answer,(i,p) in zip(Y_test[:100],enumerate(pred)):
    print(i,answer,p['result'])

