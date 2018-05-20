import tensorflow as tf

def infer(conf, x, train, regularizer):
    a = x
    for i in range(len(conf.conv_layers)):
        layer = conf.conv_layers[i]
        layer_name = "layer" + str(i+1)
        with tf.variable_scope(layer_name):
            w = tf.get_variable(
                "w",
                shape=layer.conv_shape,
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b = tf.get_variable(
                "b",
                shape=(layer.conv_shape[-1],),
                initializer=tf.constant_initializer(0.1)
            )
            conv = tf.nn.conv2d(a, w, strides=layer.conv_strides, padding='SAME')
            a = tf.nn.relu(tf.nn.bias_add(conv, b))
            a = tf.nn.max_pool(a, ksize=layer.pool_shape, strides=layer.pool_strides,padding='SAME')
    conv_out_shape = a.get_shape().as_list()
    flatten_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]
    a = tf.reshape(tensor=a, shape=[-1,flatten_size])
    prev_layer = flatten_size
    for i in range(len(conf.fc_layers)):
        layer = conf.fc_layers[i]
        layer_name = "layer" + str(len(conf.conv_layers) + i + 1)
        with tf.variable_scope(layer_name):
            w = tf.get_variable(
                "w",
                shape=(prev_layer, layer),
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b = tf.get_variable(
                "b",
                shape=(layer,),
                initializer=tf.constant_initializer(0.0)
            )
            if regularizer != None:
                tf.add_to_collection("loss",regularizer(w))
            logit = tf.matmul(a, w) + b
            if i + 1 != len(conf.fc_layers):
                a = tf.nn.relu(logit)
                if train: a = tf.nn.dropout(a, keep_prob=0.8)
        prev_layer = layer
    return logit