import tensorflow as tf
from typing import Dict


class BasePipeline:
    def __init__(self):
        self.iterator_init_ops: Dict[tf.estimator.ModeKeys, tf.Operation] = {
            tf.estimator.ModeKeys.TRAIN: None,
            tf.estimator.ModeKeys.EVAL: None
        }
        self.iterator: tf.data.Iterator = None

    def get_iterator(self) -> tf.data.Iterator:
        if self.iterator is None:
            self.iterator = self._construct_iterator()
        return self.iterator

    def _construct_iterator(self) -> tf.data.Iterator:
        raise NotImplementedError()

    def _get_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        if self.iterator_init_ops[mode] is None:
            self.iterator_init_ops[mode] = self._construct_init_op(mode)
        return self.iterator_init_ops[mode]

    def _construct_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        raise NotImplementedError()

    def switch_to(self, mode: tf.estimator.ModeKeys, sess: tf.Session = None) -> None:
        if sess is None:
            sess = tf.get_default_session()
        init_op = self._get_init_op(mode)
        sess.run(init_op)
