import tensorflow as tf

x = tf.Variable(1, dtype=tf.int32)
y = tf.Variable(2, dtype=tf.int32)

for _ in range(5):
    y.assign_add(1)
    out = x * y
    print(out)
    tf.print(out)
