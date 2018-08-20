import tensorflow as tf
import data.tiny_imagenet as data
from util.weight_decay import add_wd

NAME = "inception_transfer_001"


def graph(transfer_values, is_training, weight_decay):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # simple 2-layer graph (TODO: regularization, ...)
        fc1 = add_wd(tf.layers.dense(transfer_values, units=1024, name="classifier/fc1"), weight_decay)
        fc1 = tf.nn.relu(fc1)

        logits = add_wd(tf.layers.dense(fc1, units=data.NUM_CLASSES, name="classifier/logits"), weight_decay)
        softmax = tf.nn.softmax(logits, axis=1, name='classifier/softmax')

        return logits, softmax


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

    tf.add_to_collection('LOSSES', cross_entropy_loss)
    total_loss = tf.add_n(tf.get_collection('LOSSES'), name='total_loss')  # includes weight decay loss terms
    return total_loss


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='accuracy')

