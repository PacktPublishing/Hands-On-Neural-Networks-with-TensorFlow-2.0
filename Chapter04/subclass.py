import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(
            units=64, activation=tf.nn.elu, name="fc1")
        self.dense_2 = f.keras.layers.Dense(
            units=64, activation=tf.nn.elu, name="fc2")
        self.output = f.keras.layers.Dense(units=1, name="G")

    def call(self, inputs):
        # Build the model in functional style here
        # and return the output tensor
        net = self.dense_1(inputs)
        net = self.dense_2(net)
        net = self.output(net)
        return net
