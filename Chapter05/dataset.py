import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices({
    "a":
    tf.random.uniform([4]),
    "b":
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)
})
for value in dataset:
    print(value["a"])


def noise():
    while True:
        yield tf.random.uniform((100,))


dataset = tf.data.Dataset.from_generator(noise, (tf.float32))
buffer_size = 10
batch_size = 32
dataset = dataset.map(lambda x: x + 10).shuffle(buffer_size).batch(batch_size)
for idx, noise in enumerate(dataset):
    if idx == 2:
        break
    print(idx)
    print(noise.shape)
