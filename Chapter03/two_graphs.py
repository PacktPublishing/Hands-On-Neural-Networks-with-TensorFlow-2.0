import tensorflow as tf
g1 = tf.Graph()
g2 = tf.Graph()
with g1.as_default():
    A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x = tf.constant([[0, 10], [0, 0.5]])
    b = tf.constant([[1, -1]], dtype=tf.float32)
    y = tf.add(tf.matmul(A, x), b, name="result")

with g2.as_default():
    with tf.name_scope("scope_a"):
        x = tf.constant(1, name="x")
        print(x)
    with tf.name_scope("scope_b"):
        x = tf.constant(10, name="x")
        print(x)
    y = tf.constant(12)
    z = x * y
writer = tf.summary.FileWriter("log/two_graphs/g1", g1)
writer = tf.summary.FileWriter("log/two_graphs/g2", g2)
writer.close()
