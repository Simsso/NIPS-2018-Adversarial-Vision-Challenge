"""
Modified version of Mahmoud Gemy's code at
https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/base/base_model.py
"""

import tensorflow as tf


class BaseModel:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

        # attributes needed for global_step and global_epoch
        self.current_epoch: tf.Tensor = None
        self.increment_current_epoch: tf.Tensor = None
        self.global_step: tf.Tensor = None
        self.increment_global_step: tf.Tensor = None

        self.accuracy: tf.Tensor = None
        self.loss: tf.Tensor = None

        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_current_epoch()

        # save attribute
        self.saver: tf.train.Saver = None

        self.build_model()
        self.init_loss()
        self.init_accuracy()
        self.init_saver()

    def save(self, sess: tf.Session) -> None:
        """Saves the checkpoint in the path defined in the config file"""
        tf.logging.info("Saving model...")
        self.saver.save(sess, self.checkpoint_dir, self.global_step)
        tf.logging.info("Model saved")

    def load(self, sess: tf.Session) -> None:
        """Load latest checkpoint from the experiment path defined in the config file"""
        latest_checkpoint = self.checkpoint_dir #tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            tf.logging.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            tf.logging.info("Model loaded")
        else:
            tf.logging.warning("No checkpoint found")

    def init_current_epoch(self) -> None:
        """Initialize a TensorFlow variable to use it as epoch counter"""
        with tf.variable_scope('current_epoch'):
            self.current_epoch = tf.get_variable('current_epoch', shape=(), dtype=tf.int32, trainable=False,
                                                 initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.increment_current_epoch = tf.assign(self.current_epoch, self.current_epoch + 1)

    def init_global_step(self) -> None:
        """Initialize a TensorFlow variable to use it as global step counter"""
        with tf.variable_scope('global_step'):
            self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int32, trainable=False,
                                               initializer=tf.constant_initializer(0, dtype=tf.int32))
            self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    def init_loss(self) -> None:
        raise NotImplementedError

    def init_accuracy(self) -> None:
        raise NotImplementedError

    def init_saver(self) -> None:
        raise NotImplementedError

    def build_model(self) -> None:
        raise NotImplementedError
