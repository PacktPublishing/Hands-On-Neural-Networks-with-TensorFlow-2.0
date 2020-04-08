import tensorflow_datasets as tfds

# See available datasets
print(tfds.list_builders())
# Construct 2 tf.data.Dataset objects
# The training dataset and the test dataset
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
builder = tfds.builder("mnist")
print(builder.info)
