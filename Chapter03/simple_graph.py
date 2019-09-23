import tensorflow as tf

# Build your graph.
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result")

with tf.Session() as sess:
    print(sess.run(y))
writer = tf.summary.FileWriter("log/matmul", tf.get_default_graph())
writer.close()
