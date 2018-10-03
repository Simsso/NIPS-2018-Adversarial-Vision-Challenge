import logging
import tensorflow as tf
from tensorflow.python.platform import tf_logging
import sys


def init():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf_logger = logging.getLogger('tensorflow')
    handler = logging.StreamHandler(sys.stdout) # create stdout handler
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tf_logger.handlers = [handler] # redirect tf.logging to stdout instead of stderr
    tf_logging.propagate = False
