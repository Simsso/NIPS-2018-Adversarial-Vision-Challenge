"""
Modified version of Mahmoud Gemy's code at
https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
"""

import tensorflow as tf
from typing import Optional

from resnet_base.util.logger.factory import LoggerFactory

tf.flags.DEFINE_string("global_checkpoint", "", "Checkpoint path of all weights.")
FLAGS = tf.flags.FLAGS


class BaseModel:
    def __init__(self, logger_factory: LoggerFactory = None):

        if logger_factory is None:
            logger_factory = LoggerFactory()
        self.logger_factory = logger_factory

        # attributes needed for global_step and global_epoch
        self.current_epoch = None
        self.increment_current_epoch = None
        self.global_step = None
        self.increment_global_step = None

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_current_epoch()

        # saver attribute
        self.saver = None

    def post_build_init(self):
        self.init_saver()

    def init_current_epoch(self) -> None:
        """Initialize a TensorFlow variable to use it as epoch counter"""
        with tf.variable_scope('meta'):
            self.current_epoch = tf.get_variable('current_epoch', shape=(), dtype=tf.int32, trainable=False,
                                                 initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.increment_current_epoch = tf.assign(self.current_epoch, self.current_epoch + 1)

    def init_global_step(self) -> None:
        """Initialize a TensorFlow variable to use it as global step counter"""
        with tf.variable_scope('meta'):
            self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int32, trainable=False,
                                               initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    def init_saver(self) -> None:
        raise NotImplementedError

    def save(self, sess: tf.Session) -> None:
        BaseModel._save_to_path(sess, self.saver, self.global_step, FLAGS.global_checkpoint)

    def restore(self, sess: tf.Session) -> None:
        BaseModel._restore_checkpoint(self.saver, sess, FLAGS.global_checkpoint)

    @staticmethod
    def _restore_checkpoint(saver: tf.train.Saver, sess: tf.Session, path: Optional[str] = None):
        if path and saver:
            saver.restore(sess, path)
            tf.logging.info("Model loaded from {}".format(path))

    @staticmethod
    def _create_saver(collection_name: str) -> Optional[tf.train.Saver]:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, collection_name)
        if var_list:
            return tf.train.Saver(var_list=var_list)
        return None

    @staticmethod
    def _save_to_path(sess: tf.Session, saver: Optional[tf.train.Saver], global_step: tf.Tensor, path: Optional[str]):
        if saver and path:
            tf.logging.info("Saving model...")
            saver.save(sess, path, global_step=global_step)
            tf.logging.info("Model saved to {}".format(path))
