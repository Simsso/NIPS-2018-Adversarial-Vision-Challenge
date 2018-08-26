import re
import tensorflow as tf


def activation_summary(x):
    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py#L80
    tensor_name = re.sub('[0-9]*/', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def weight_summary(x):
    tf.summary.histogram(x.op.name, x)


def __get_trainable_vars(exclude=None):
    if exclude is None:
        exclude = []
    return [v for v in tf.trainable_variables() if v not in exclude]


def weight_summary_for_all():
    list(map(weight_summary, __get_trainable_vars()))
