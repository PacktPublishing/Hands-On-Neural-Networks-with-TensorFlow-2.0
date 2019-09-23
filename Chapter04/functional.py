import tensorflow as tf

input_shape = (100,)
inputs = tf.keras.layers.Input(input_shape)
net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
net = tf.keras.layers.Dense(units=1, name="G")(net)
model = tf.keras.Model(inputs=inputs, outputs=net)
