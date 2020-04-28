import tensorflow as tf


@tf.function
def f():
    for i in range(10):
        print(i)


f()
f()
