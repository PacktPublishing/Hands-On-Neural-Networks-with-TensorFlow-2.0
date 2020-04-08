import tensorflow as tf

x = tf.Variable(4.0)
y = tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
    z = x + y
    w = tf.pow(x, 2)
dz_dy = tape.gradient(z, y)
dz_dx = tape.gradient(z, x)
dw_dx = tape.gradient(w, x)
print(dz_dy, dz_dx, dw_dx)
# Release the resources
del tape
