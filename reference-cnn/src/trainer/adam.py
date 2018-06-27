import tensorflow as tf

LEARNING_RATE = 0.01


def train(loss):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    return optimizer.minimize(loss)