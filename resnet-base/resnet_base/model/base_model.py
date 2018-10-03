"""
Modified version of Mahmoud Gemy's code at
https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
"""

import tensorflow as tf
from typing import Optional
import os

from resnet_base.util.logger.factory import LoggerFactory

tf.flags.DEFINE_string("save_dir", "", "Checkpoint directory of the complete graph's variables. Used both to \
                                              restore (if available) and to save the model.")
tf.flags.DEFINE_string("name", "model", "The name of the model (may contain hyperparameter information), used when \
                                        saving the model.")

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
        BaseModel._save_to_path(sess, self.saver, self.global_step, path=FLAGS.save_dir)

    def restore(self, sess: tf.Session) -> None:
        BaseModel._restore_checkpoint(self.saver, sess, path=FLAGS.save_dir)

    @staticmethod
    def _restore_checkpoint(saver: tf.train.Saver, sess: tf.Session, path: Optional[str] = None):
        if path and saver:
            # if a directory is given instead of a path, try to find a checkpoint file there
            checkpoint_file = tf.train.latest_checkpoint(path) if os.path.isdir(path) else path

            if checkpoint_file and tf.train.checkpoint_exists(checkpoint_file):
                saver.restore(sess, checkpoint_file)
                tf.logging.info("Model loaded from {}".format(checkpoint_file))
            else:
                tf.logging.info("No valid checkpoint has been found at {}. Ignoring.".format(path))

    @staticmethod
    def _create_saver(collection_name: str) -> Optional[tf.train.Saver]:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, collection_name)
        if var_list:
            return tf.train.Saver(var_list=var_list)
        return None

    @staticmethod
    def _save_to_path(sess: tf.Session, saver: Optional[tf.train.Saver], global_step: tf.Tensor, path: Optional[str]):
        if saver and path:
            if os.path.isdir(path):
                path = os.path.join(path, FLAGS.name + ".ckpt")
            save_path = saver.save(sess, path, global_step=global_step)
            tf.logging.info("Model saved to {}".format(save_path))
