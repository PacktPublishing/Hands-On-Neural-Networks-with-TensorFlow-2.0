import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

n_classes = 10
model = tf.keras.Sequential([
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

model.summary()


def train_dataset(batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    half = test_x.shape[0] // 2
    input_x, input_y = train_x, train_y

    def scale_fn(image, label):
        return (
            tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0, label

    dataset = tf.data.Dataset.from_tensor_slices((tf.expand_dims(
        input_x, -1), tf.expand_dims(input_y, -1))).map(scale_fn)

    dataset = dataset.cache().repeat(num_epochs)
    dataset = dataset.shuffle(batch_size)

    return dataset.batch(batch_size).prefetch(1)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_dataset(num_epochs=10))
