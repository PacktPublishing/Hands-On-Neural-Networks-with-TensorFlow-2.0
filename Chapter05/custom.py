import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def train_dataset(batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    input_x, input_y = train_x, train_y

    def scale_fn(image, label):
        return (
            tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0, label

    dataset = tf.data.Dataset.from_tensor_slices((tf.expand_dims(
        input_x, -1), tf.expand_dims(input_y, -1))).map(scale_fn)

    dataset = dataset.cache().repeat(num_epochs)
    dataset = dataset.shuffle(batch_size)

    return dataset.batch(batch_size).prefetch(1)


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


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    dataset = train_dataset(num_epochs=10)

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)
    accuracy = tf.metrics.Accuracy()

    # Train step function
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy_value = accuracy(labels, tf.argmax(logits, -1))
        return loss_value, accuracy_value

    @tf.function
    def loop():
        for features, labels in dataset:
            loss_value, accuracy_value = train_step(features, labels)
            if tf.equal(tf.mod(step, 10), 0):
                tf.print(step, ": ", loss_value, " - accuracy: ",
                         accuracy_value)

    loop()


if __name__ == "__main__":
    train()
