import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def make_model(n_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes)
    ])


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = test_x / 255. * 2 - 1
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)
    accuracy = tf.metrics.Accuracy()

    # Train step function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        # TODO: apply gradient clipping here
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy_value = accuracy(labels, tf.argmax(logits, -1))
        return loss_value, accuracy_value

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    for epoch in range(epochs):
        for t in range(nr_batches_train):
            start_from = t * batch_size
            to = (t + 1) * batch_size

            features, labels = train_x[start_from:to], train_y[start_from:to]

            loss_value, accuracy_value = train_step(features, labels)
            if t % 10 == 0:
                print(
                    f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}"
                )
        print(f"Epoch {epoch} terminated")


if __name__ == "__main__":
    train()
