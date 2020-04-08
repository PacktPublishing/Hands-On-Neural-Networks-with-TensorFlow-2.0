import tensorflow as tf
import numpy as np

A = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.Variable([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result")

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter("log/matmul", tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    sess.run(init)
    A_value, x_value, b_value = sess.run([A, x, b])
    y_value = sess.run(y)

    # Overwrite
    y_new = sess.run(y, feed_dict={b: np.zeros((1, 2))})

print(f"A: {A_value}\nx: {x_value}\nb: {b_value}\n\ny: {y_value}")
print(f"y_new: {y_new}")
