import tensorflow as tf
import data.tiny_imagenet as data

NAME = "inception_transfer_001"


def graph(transfer_values, is_training, wd):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # simple 2-layer graph (TODO: regularization, ...)
        fc1 = tf.layers.dense(transfer_values, units=1024, name="classifier/fc1")
        fc1 = tf.nn.relu(fc1)

        logits = tf.layers.dense(fc1, units=data.NUM_CLASSES, name="classifier/logits")
        softmax = tf.nn.softmax(logits, axis=1, name='classifier/softmax')

        return logits, softmax


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    loss_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    return loss_mean


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='accuracy')

