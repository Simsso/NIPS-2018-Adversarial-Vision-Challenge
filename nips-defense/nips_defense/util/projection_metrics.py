import tensorflow as tf
from typing import Tuple


def projection_identity_accuracy(identity_mask: tf.Tensor, logits: tf.Tensor, projection_labels: tf.Tensor,
                                 labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates the classification accuracy for both the identity-mapped inputs and the projected inputs and logs
    them (based on a k-NN cosine-similarity vector quantization layer).
    :param identity_mask: a boolean mask that specifies which of the inputs are identity-mapped and which are not,
           shape [batch]
    :param logits: the logits output of the network, shape [batch, num_classes]
    :param projection_labels: the labels that the projected inputs are assigned by the quantization, shape [batch]
    :param labels: the actual ground-truth labels of the batch, shape [batch]
    :return: a tuple of two tensors:
                - accuracy_projection: the classification accuracy of the projected inputs
                - accuracy_identity: the classification accuracy of the identity-mapped inputs
    """
    labels = tf.cast(labels, tf.int64)
    identity_softmax = tf.nn.softmax(logits)

    # adding .001 so we don't divide by zero
    num_identity_mapped = tf.reduce_sum(tf.cast(identity_mask, dtype=tf.float32)) + .001
    num_projected = tf.reduce_sum(tf.cast(tf.logical_not(identity_mask), dtype=tf.float32)) + .001

    correct_identity = tf.equal(tf.argmax(identity_softmax, axis=1), labels)
    # only include the correctly classified inputs that are identity-mapped
    correct_identity = tf.logical_and(correct_identity, identity_mask)
    accuracy_identity = tf.reduce_sum(tf.cast(correct_identity, dtype=tf.float32)) / num_identity_mapped

    correct_projection = tf.equal(tf.cast(projection_labels, tf.int64), labels)
    # only include the correctly classified inputs that are *not* identity-mapped, i.e. projected
    correct_projection = tf.logical_and(correct_projection, tf.logical_not(identity_mask))
    accuracy_projection = tf.reduce_sum(tf.cast(correct_projection, dtype=tf.float32)) / num_projected

    return accuracy_projection, accuracy_identity
