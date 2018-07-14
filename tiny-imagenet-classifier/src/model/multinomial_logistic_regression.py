import tensorflow as tf
import data.tiny_imagenet as dataset


def model(x):
    end_points = {}  # dictionary of relevant model ops

    logits = tf.layers.dense(x, dataset.NUM_CLASSES, use_bias=True, name='dense1')
    softmax = tf.nn.softmax(logits, axis=1)

    end_points['logits'] = logits
    end_points['softmax'] = softmax

    return end_points


def loss(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='cross_entropy_loss')


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='acc')

