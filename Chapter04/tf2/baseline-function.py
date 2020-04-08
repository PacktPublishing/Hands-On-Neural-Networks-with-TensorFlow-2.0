import tensorflow as tf


def multiply(x, y):
    """Matrix multiplication.
    Note: it requires the input shape of both input to match.
    Args:
        x: tf.Tensor a matrix
        y: tf.Tensor a matrix
    Returns:
        The matrix multiplcation x @ y
    """

    assert x.shape == y.shape
    return tf.matmul(x, y)


def add(x, y):
    """Add two tensors.
    Args:
        x: the left hand operand.
        y: the right hand operand. It should be compatible with x.
    Returns:
        x + y
    """
    return x + y


def main():
    """Main program."""
    A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x = tf.constant([[0, 10], [0, 0.5]])
    b = tf.constant([[1, -1]], dtype=tf.float32)

    z = multiply(A, x)
    y = add(z, b)
    print(y)


if __name__ == "__main__":
    main()
