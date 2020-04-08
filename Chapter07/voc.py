import tensorflow as tf
import tensorflow_hub as hub

import tensorflow_datasets as tfds

# Train, test, and validation are datasets for object detection: multiple objects per image.
(train, test, validation), info = tfds.load(
    "voc/2007", split=["train", "test", "validation"], with_info=True
)

# Create a subset of the dataset by filtering the elements: we are interested
# in creating a dataset for object detetion and classification that is a dataset
# of images with a single object annotated.
def filter(dataset):
    return dataset.filter(lambda row: tf.equal(tf.shape(row["objects"]["label"])[0], 1))


train, test, validation = filter(train), filter(test), filter(validation)


# Input layer
inputs = tf.keras.layers.Input(shape=(299, 299, 3))

# Feature extractor
net = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
    output_shape=[2048],
    trainable=False,
)(inputs)

# Regression head
regression_head = tf.keras.layers.Dense(512)(net)
regression_head = tf.keras.layers.ReLU()(regression_head)
coordinates = tf.keras.layers.Dense(4, use_bias=False)(regression_head)

regressor = tf.keras.Model(inputs=inputs, outputs=coordinates)

# Classification head
classification_head = tf.keras.layers.Dense(1024)(net)
classification_head = tf.keras.layers.ReLU()(classification_head)
classification_head = tf.keras.layers.Dense(128)(net)
classification_head = tf.keras.layers.ReLU()(classification_head)
num_classes = 20
classification_head = tf.keras.layers.Dense(num_classes, use_bias=False)(
    classification_head
)

model = tf.keras.Model(inputs=inputs, outputs=[coordinates, classification_head])


def prepare(dataset):
    def _fn(row):
        row["image"] = tf.image.convert_image_dtype(row["image"], tf.float32)
        row["image"] = tf.image.resize(row["image"], (299, 299))
        return row

    return dataset.map(_fn)


train, test, validation = prepare(train), prepare(test), prepare(validation)

# First option -> this requires to call the loss l2, taking care of squeezing the input
# l2 = tf.losses.MeanSquaredError()

# Second option, it is the loss function iself that squeezes the input
def l2(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - tf.squeeze(y_true, axis=[1])))


precision_metric = tf.metrics.Precision()


def iou(pred_box, gt_box, h, w):
    """
    Compute IoU between detect box and gt boxes
    Args:
        pred_box: shape (4, ):  y_min, x_min, y_max, x_max - predicted box
        gt_boxes: shape (n, 4): y_min, x_min, y_max, x_max - ground truth
        h: image height
        w: image width
    """

    # Transpose the coordinates from y_min, x_min, y_max, x_max
    # In absolute coordinates to x_min, y_min, x_max, y_max
    # in pixel coordinates
    def _swap(box):
        return tf.stack([box[1] * w, box[0] * h, box[3] * w, box[2] * h])

    pred_box = _swap(pred_box)
    gt_box = _swap(gt_box)

    box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    xx1 = tf.maximum(pred_box[0], gt_box[0])
    yy1 = tf.maximum(pred_box[1], gt_box[1])
    xx2 = tf.minimum(pred_box[2], gt_box[2])
    yy2 = tf.minimum(pred_box[3], gt_box[3])

    # compute the width and height of the bounding box
    w = tf.maximum(0, xx2 - xx1)
    h = tf.maximum(0, yy2 - yy1)

    inter = w * h
    return inter / (box_area + area - inter)


threshold = 0.75


def draw(dataset, regressor, step):
    with tf.device("/CPU:0"):
        row = next(iter(dataset.take(3).batch(3)))
        images = row["image"]
        obj = row["objects"]
        boxes = regressor(images)

        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(boxes, (-1, 1, 4)), colors=[(1.0, 0, 0, 0)]
        )
        images = tf.image.draw_bounding_boxes(
            images=images,
            boxes=tf.reshape(obj["bbox"], (-1, 1, 4)),
            colors=[(1.0, 0, 0, 0)],
        )
        tf.summary.image("images", images, step=step)

        true_labels, predicted_labels = [], []
        for idx, predicted_box in enumerate(boxes):
            iou_value = iou(predicted_box, tf.squeeze(obj["bbox"][idx]), 299, 299)
            true_labels.append(1)
            predicted_labels.append(1 if iou_value >= threshold else 0)

        precision_metric.update_state(true_labels, predicted_labels)
        tf.summary.scalar("precision", precision_metric.result(), step=step)
        tf.print(precision_metric.result())


optimizer = tf.optimizers.Adam()
epochs = 10
batch_size = 3

global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

train_writer, validation_writer = (
    tf.summary.create_file_writer("log/train"),
    tf.summary.create_file_writer("log/validation"),
)
with validation_writer.as_default():
    draw(validation, regressor, global_step)


@tf.function
def train_step(image, coordinates):
    with tf.GradientTape() as tape:
        loss = l2(coordinates, regressor(image))
    gradients = tape.gradient(loss, regressor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, regressor.trainable_variables))
    return loss


train_batches = train.cache().batch(batch_size).prefetch(1)
with train_writer.as_default():
    for _ in tf.range(epochs):
        for batch in train_batches:
            obj = batch["objects"]
            coordinates = obj["bbox"]
            loss = train_step(batch["image"], coordinates)
            tf.summary.scalar("loss", loss, step=global_step)
            global_step.assign_add(1)
            if tf.equal(tf.math.mod(global_step, 10), 0):
                tf.print("step ", global_step, " loss: ", loss)
                with validation_writer.as_default():
                    draw(validation, regressor, global_step)
                with train_writer.as_default():
                    draw(train, regressor, global_step)
    # Clean the metrics at the end of every epoch
    precision_metric.reset()
