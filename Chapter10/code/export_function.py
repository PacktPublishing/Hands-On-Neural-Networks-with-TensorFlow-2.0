import tensorflow as tf


class Wrapper(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ]
    )
    def pow(self, x, y):
        return tf.math.pow(x, y), tf.math.pow(y, x)


obj = Wrapper()
tf.saved_model.save(obj, "/tmp/pow/1")

path = "/tmp/pow/1"

imported = tf.saved_model.load(path)

imported.signatures["serving_default"](x=tf.constant(2.0), y=tf.constant(5.0))

import tensorflow as tf


class Wrapper(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
        ]
    )
    def pow(self, x, y):
        return {"pow_x_y": tf.math.pow(x, y), "pow_y_x": tf.math.pow(y, x)}


obj = Wrapper()
tf.saved_model.save(obj, "/tmp/pow/1")

path = "/tmp/pow/1"

imported = tf.saved_model.load(path)
imported.signatures["serving_default"](x=tf.constant(2.0), y=tf.constant(5.0))
