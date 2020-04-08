import tensorflow as tf

x = tf.Variable(1, dtype=tf.int32)
y = tf.Variable(2, dtype=tf.int32)

assign_op = tf.assign_add(y, 1)

with tf.control_dependencies([assign_op]):
    out = x * y

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(5):
        print(sess.run(out))
