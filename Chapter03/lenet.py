import tensorflow as tf


def define_cnn(x, n_classes, reuse, is_training):
    """Defines a convolutional neural network for classification.
    Args:
        x: a batch of images: 4D tensor.
        n_classes: the number of classes, hence, the number of output neurons.
        reuse: the `tf.variable_scope` reuse parameter.
        is_training: boolean variable that indicates if the model is in training.
    Returns:
        The output layer.
    """
    with tf.variable_scope('cnn', reuse=reuse):
        # Convolution Layer with 32 learneable filters 5x5 each
        # followed by max-pool operation that halves the spatial extent.
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 learneable filters 3x3 each.
        # As above, max pooling to halve.
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1D vector so we can use a fully connected layer.
        # Please note how the new shape is computed and how the negative dimension
        # in the batch size position.
        shape = (
            -1,
            conv2.shape[1].value * conv2.shape[2].value * conv2.shape[3].value)
        fc1 = tf.reshape(conv2, shape)

        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply (inverted) dropout when in training phase.
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        # Prediction: linear neurons
        out = tf.layers.dense(fc1, n_classes)

    return out


input = tf.placeholder(tf.float32, (None, 28, 28, 1))
logits = define_cnn(input, 10, reuse=False, is_training=True)
writer = tf.summary.FileWriter("log/cnn", tf.get_default_graph())
writer.close()
