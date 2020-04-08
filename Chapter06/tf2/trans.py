import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


dataset, info = tfds.load("tf_flowers", with_info=True)
print(info)
dataset = dataset["train"]
tot = 3670

train_set_size = tot // 2
validation_set_size = tot - train_set_size - train_set_size // 2
test_set_size = tot - train_set_size - validation_set_size


print("train set size: ", train_set_size)
print("validation set size: ", validation_set_size)
print("test set size: ", test_set_size)

train, test, validation = (
    dataset.take(train_set_size),
    dataset.skip(train_set_size).take(validation_set_size),
    dataset.skip(train_set_size + validation_set_size).take(test_set_size),
)


def to_float_image(example):
    example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
    return example


def resize(example):
    example["image"] = tf.image.resize(example["image"], (299, 299))
    return example


train = train.map(to_float_image).map(resize)
validation = validation.map(to_float_image).map(resize)
test = test.map(to_float_image).map(resize)

num_classes = 5

model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
            output_shape=[2048],
            trainable=False,
        ),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(num_classes),  # linear
    ]
)

# Training utilities
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
step = tf.Variable(1, name="global_step", trainable=False)
optimizer = tf.optimizers.Adam(1e-3)

train_summary_writer = tf.summary.create_file_writer("./log/transfer/train")
validation_summary_writer = tf.summary.create_file_writer("./log/transfer/validation")

# Metrics
accuracy = tf.metrics.Accuracy()
mean_loss = tf.metrics.Mean(name="loss")


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss(labels, logits)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    step.assign_add(1)

    accuracy.update_state(labels, tf.argmax(logits, -1))
    return loss_value


# Configure the training set to use batches and prefetch
train = train.batch(32).prefetch(1)
validation = validation.batch(32).prefetch(1)
test = test.batch(32).prefetch(1)

num_epochs = 10
for epoch in range(num_epochs):

    for example in train:
        image, label = example["image"], example["label"]
        loss_value = train_step(image, label)
        mean_loss.update_state(loss_value)

        if tf.equal(tf.math.mod(step, 10), 0):
            tf.print(
                step, " loss: ", mean_loss.result(), " acccuracy: ", accuracy.result()
            )
            mean_loss.reset_states()
            accuracy.reset_states()

    # Epoch ended, measure performance on validation set
    tf.print("## VALIDATION - ", epoch)
    accuracy.reset_states()
    for example in validation:
        image, label = example["image"], example["label"]
        logits = model(image)
        accuracy.update_state(label, tf.argmax(logits, -1))
    tf.print("accuracy: ", accuracy.result())
    accuracy.reset_states()
