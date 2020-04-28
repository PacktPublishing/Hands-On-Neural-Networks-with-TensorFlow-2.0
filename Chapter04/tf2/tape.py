import tensorflow as tf

x = tf.constant(4.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.pow(x, 2)
# Will compute 8 = 2*x, x = 8
dy_dx = tape.gradient(y, x)
print(dy_dx)
