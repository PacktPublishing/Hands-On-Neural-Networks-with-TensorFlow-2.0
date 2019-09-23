import tensorflow as tf
import tensorflow_datasets as tfds
import math
import os


def downsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                depth, 3, strides=2, padding="same", kernel_initializer="he_normal"
            ),
            tf.keras.layers.LeakyReLU(),
        ]
    )


def upsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(
                depth, 3, strides=2, padding="same", kernel_initializer="he_normal"
            ),
            tf.keras.layers.ReLU(),
        ]
    )

def get_unet(input_size=(256, 256, 3), num_classes=21):

    # Downsample from 256x256 to 4x4, while adding depth
    # using powers of 2, startin from 2**5. Cap to 512.
    encoders = []
    for i in range(2, int(math.log2(256))):
        depth = 2 ** (i + 5)
        if depth > 512:
            depth = 512
        encoders.append(downsample(depth=depth))

    # Upsample from 4x4 to 256x256, reducing the depth
    decoders = []
    for i in reversed(range(2, int(math.log2(256)))):
        depth = 2 ** (i + 5)
        if depth < 32:
            depth = 32
        if depth > 512:
            depth = 512
        decoders.append(upsample(depth=depth))

    # Build the model by invoking the encoder layers with the correct input
    inputs = tf.keras.layers.Input(input_size)
    concat = tf.keras.layers.Concatenate()

    x = inputs
    # Encoder: downsample loop
    skips = []
    for conv in encoders:
        x = conv(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder: input + skip connection
    for deconv, skip in zip(decoders, skips):
        x = deconv(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Add the last layer on top and define the model
    last = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding="same", kernel_initializer="he_normal"
    )

    outputs = last(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

LUT = {
    (0, 0, 0): 0, # background
    (128, 0, 0): 1, # aeroplane
    (0, 128, 0): 2, # bicycle
    (128, 128, 0): 3, # bird
    (0, 0, 128): 4, # boat
    (128, 0, 128): 5, # bottle
    (0, 128, 128): 6, # bus
    (128, 128, 128): 7, # car
    (64, 0, 0): 8, # cat
    (192, 0, 0): 9, # chair
    (64, 128, 0): 10, # cow
    (192, 128, 0): 11, # diningtable
    (64, 0, 128): 12, # dog
    (192, 0, 128): 13, # horse
    (64, 128, 128): 14, # motorbike
    (192, 128, 128): 15, # person
    (0, 64, 0): 16, # pottedplant
    (128, 64, 0): 17, # sheep
    (0, 192, 0): 18, # sofa
    (128, 192, 0): 19, # train
    (0, 64, 128): 20, # tvmonitor
    (255, 255, 255): 21, # undefined / don't care
}


class Voc2007Semantic(tfds.image.Voc2007): 
    """Pasval VOC 2007 - semantic segmentation.""" 
 
    VERSION = tfds.core.Version("0.1.0")
    def _info(self):
        parent_info = tfds.image.Voc2007().info
        return tfds.core.DatasetInfo(
            builder=self,
            description=parent_info.description,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "image/filename": tfds.features.Text(),
                    "label": tfds.features.Image(shape=(None, None, 1)),
                }
            ),
            urls=parent_info.urls,
            citation=parent_info.citation,
        )

      
    def _generate_examples(self, data_path, set_name):
        set_filepath = os.path.join(
            data_path,
            "VOCdevkit/VOC2007/ImageSets/Segmentation/{}.txt".format(set_name),
        )
        with tf.io.gfile.GFile(set_filepath, "r") as f:
            for line in f:
                image_id = line.strip()

                image_filepath = os.path.join(
                    data_path, "VOCdevkit", "VOC2007", "JPEGImages", f"{image_id}.jpg"
                )
                label_filepath = os.path.join(
                    data_path,
                    "VOCdevkit",
                    "VOC2007",
                    "SegmentationClass",
                    f"{image_id}.png",
                )

                if not tf.io.gfile.exists(label_filepath):
                    continue

                label_rgb = tf.image.decode_image(
                    tf.io.read_file(label_filepath), channels=3
                )

                label = tf.Variable(
                    tf.expand_dims(
                        tf.zeros(shape=tf.shape(label_rgb)[:-1], dtype=tf.uint8), -1
                    )
                )

                for color, label_id in LUT.items():
                    match = tf.reduce_all(tf.equal(label_rgb, color), axis=[2])
                    labeled = tf.expand_dims(tf.cast(match, tf.uint8), axis=-1)
                    label.assign_add(labeled * label_id)

                colored = tf.not_equal(tf.reduce_sum(label), tf.constant(0, tf.uint8))
                # Certain labels have wrong RGB values
                if not colored.numpy():
                    tf.print("error parsing: ", label_filepath)
                    continue
                
                yield {
                    # Declaring in _info "image" as a tfds.feature.Image
                    # we can use both an image or a string. If a string is detected
                    # it is supposed to be the image path and tfds take care of the
                    # reading process.
                    "image": image_filepath,
                    "image/filename": f"{image_id}.jpg",
                    "label": label.numpy(),
                }

print(tfds.list_builders())
dataset, info = tfds.load("voc2007_semantic", with_info=True)

train_set = dataset["train"]

def resize_and_scale(row):
    # Resize and convert to float, [0,1] range
    row["image"] = tf.image.convert_image_dtype(
        tf.image.resize(
            row["image"],
            (256,256),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.float32)
    # Resize, cast to int64 since it is a supported label type
    row["label"] = tf.cast(
        tf.image.resize(
            row["label"],
            (256,256),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.int64)
    return row
  
def to_pair(row):
    return row["image"], row["label"]
batch_size= 32

train_set = train_set.map(resize_and_scale).map(to_pair)
train_set = train_set.batch(batch_size).prefetch(1)

validation_set = dataset["validation"].map(resize_and_scale)
validation_set = validation_set.map(to_pair).batch(batch_size)

model = get_unet()

optimizer = tf.optimizers.Adam()

checkpoint_path = "ckpt/pb.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(write_images=True)
model.compile(optimizer=optimizer,
              #loss=lambda y_true, y_pred: tf.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred) + tf.losses.MeanAbsoluteError()(y_true, y_pred),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#, tf.metrics.MeanIoU(num_classes=21)])


num_epochs = 50
model.fit(train_set, validation_data=validation_set, epochs=num_epochs,
          callbacks=[cp_callback, tensorboard])


sample = tf.image.decode_jpeg(tf.io.read_file("me.jpg"))
sample = tf.expand_dims(tf.image.convert_image_dtype(sample, tf.float32), axis=[0])
sample = tf.image.resize(sample, (512,512))
pred_image = tf.squeeze(tf.argmax(model(sample), axis=-1), axis=[0])

REV_LUT = {value: key for key, value in LUT.items()}

color_image = tf.Variable(tf.zeros((512,512,3), dtype=tf.uint8))
pixels_per_label = []
for label, color in REV_LUT.items():
    match = tf.equal(pred_image, label)
    labeled = tf.expand_dims(tf.cast(match, tf.uint8), axis=-1)
    pixels_per_label.append((label, tf.math.count_nonzero(labeled)))
    labeled = tf.tile(labeled, [1,1,3])
    color_image.assign_add(labeled * color)


for label, count in pixels_per_label:
    print(label, ": ", count.numpy())

tf.io.write_file("seg.jpg", tf.io.encode_jpeg(color_image))

