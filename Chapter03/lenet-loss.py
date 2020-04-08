import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


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
    with tf.variable_scope("cnn", reuse=reuse):
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
        shape = (-1, conv2.shape[1].value * conv2.shape[2].value * conv2.shape[3].value)
        fc1 = tf.reshape(conv2, shape)

        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply (inverted) dropout when in training phase.
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        # Prediction: linear neurons
        out = tf.layers.dense(fc1, n_classes)

    return out


def train():

    input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    labels = tf.placeholder(tf.int64, (None,))
    logits = define_cnn(input, 10, reuse=False, is_training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step)

    writer = tf.summary.FileWriter("log/graph_loss", tf.get_default_graph())
    validation_summary_writer = tf.summary.FileWriter("log/graph_loss/validation")

    init_op = tf.global_variables_initializer()

    predictions = tf.argmax(logits, 1)
    # correct predictions: [BATCH_SIZE] tensor
    correct_predictions = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    loss_summary = tf.summary.scalar("loss", loss)

    ### python

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = train_x / 255.0 * 2 - 1
    train_x = np.expand_dims(train_x, -1)
    test_x = test_x / 255.0 * 2 - 1
    test_x = np.expand_dims(test_x, -1)

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    validation_accuracy = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(epochs):
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size

                loss_value, _, step = sess.run(
                    [loss, train_op, global_step],
                    feed_dict={
                        input: train_x[start_from:to],
                        labels: train_y[start_from:to],
                    },
                )
                if t % 10 == 0:
                    print(f"{step}: {loss_value}")
            print(f"Epoch {epoch} terminated: measuring metrics and logging summaries")

            saver.save(sess, "log/graph_loss/model")
            start_from = 0
            to = 128
            train_accuracy_summary, train_loss_summary = sess.run(
                [accuracy_summary, loss_summary],
                feed_dict={
                    input: train_x[start_from:to],
                    labels: train_y[start_from:to],
                },
            )

            validation_accuracy_summary, validation_accuracy_value, validation_loss_summary = sess.run(
                [accuracy_summary, accuracy, loss_summary],
                feed_dict={input: test_x[start_from:to], labels: test_y[start_from:to]},
            )

            # save values in tensorboard
            writer.add_summary(train_accuracy_summary, step)
            writer.add_summary(train_loss_summary, step)

            validation_summary_writer.add_summary(validation_accuracy_summary, step)
            validation_summary_writer.add_summary(validation_loss_summary, step)

            validation_summary_writer.flush()
            writer.flush()

            # model selection
            if validation_accuracy_value > validation_accuracy:
                validation_accuracy = validation_accuracy_value
                saver.save(sess, "log/graph_loss/best_model/best")

    writer.close()


if __name__ == "__main__":
    train()
