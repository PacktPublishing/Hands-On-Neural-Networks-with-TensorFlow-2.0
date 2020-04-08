import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def get_input_fn(mode, batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    half = test_x.shape[0] // 2
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_x, input_y = train_x, train_y
        train = True
    elif mode == tf.estimator.ModeKeys.EVAL:
        input_x, input_y = test_x[:half], test_y[:half]
        train = False
    elif mode == tf.estimator.ModeKeys.PREDICT:
        input_x, input_y = test_x[half:-1], test_y[half:-1]
        train = False
    else:
        raise ValueError("tf.estimator.ModeKeys required!")

    def scale_fn(image, label):
        return (
            (tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0,
            tf.cast(label, tf.int32),
        )

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.expand_dims(input_x, -1), tf.expand_dims(input_y, -1))
        ).map(scale_fn)
        if train:
            dataset = dataset.shuffle(10).repeat(num_epochs)
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    return input_fn


def make_model(n_classes):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes),
        ]
    )


def model_fn(features, labels, mode):
    v1 = tf.compat.v1
    model = make_model(10)
    logits = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract the predictions
        predictions = v1.argmax(logits, -1)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = v1.reduce_mean(
        v1.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=v1.squeeze(labels)
        )
    )

    global_step = v1.train.get_global_step()

    # Compute evaluation metrics.
    accuracy = v1.metrics.accuracy(
        labels=labels, predictions=v1.argmax(logits, -1), name="accuracy"
    )
    # The metrics dictionary is used by the estimator during the evaluation
    metrics = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = v1.train.AdamOptimizer(1e-4)
        train_op = opt.minimize(
            loss, var_list=model.trainable_variables, global_step=global_step
        )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    raise NotImplementedError(f"Unknown mode {mode}")


print("Every log is on Tensorboard, please run tensorboard --logidr log")
estimator = tf.estimator.Estimator(model_fn, model_dir="log")
for epoch in range(50):
    print(f"Training for the {epoch}-th epoch")
    estimator.train(get_input_fn(tf.estimator.ModeKeys.TRAIN, num_epochs=1))
    print("Evaluating...")
    estimator.evaluate(get_input_fn(tf.estimator.ModeKeys.EVAL))
