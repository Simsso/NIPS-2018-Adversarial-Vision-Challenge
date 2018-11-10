import tensorflow as tf
from typing import Dict, Optional


class BasePipeline:
    """
    A pipeline represents multiple input streams for a dataset. One contains training samples the other one validation
    samples.
    """
    def __init__(self):
        self.iterator_init_ops = {
            tf.estimator.ModeKeys.TRAIN: None,
            tf.estimator.ModeKeys.EVAL: None
        }
        self.iterator = None

    def get_iterator(self) -> tf.data.Iterator:
        """
        :return: An iterator on which get_next() can be called in order to retrieve tensors which correspond to dataset
                 samples when being evaluated.
        """
        if self.iterator is None:
            self.iterator = self._construct_iterator()
        return self.iterator

    def _construct_iterator(self) -> tf.data.Iterator:
        """
        Creates an iterator for this pipeline. Shall only be called once because the iterator is stored in an attribute.
        :return: A new iterator.
        """
        raise NotImplementedError()

    def _get_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        """
        :param mode: TRAIN (training) or EVAL (validation)
        :return: An op which switches the pipeline to serve samples of the desired kind (training or validation).
        """
        if self.iterator_init_ops[mode] is None:
            self.iterator_init_ops[mode] = self._construct_init_op(mode)
        return self.iterator_init_ops[mode]

    def _construct_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        """
        Creates a new initialization operation for the given mode key. This function is only being called once per mode
        because the op is then stored in an attribute.
        :param mode: TRAIN (training) or EVAL (validation)
        :return: Initialization operation which switches the pipeline when being evaluated.
        """
        raise NotImplementedError()

    def switch_to(self, mode: tf.estimator.ModeKeys, feed_dict: Optional[Dict] = None, sess: tf.Session = None) -> None:
        """
        Switches the input pipeline to the given mode in the given session.
        :param mode: TRAIN (training) or EVAL (validation)
        :param feed_dict: Feed dictionary that will be passed when evaluating the initialization op.
        :param sess: Session to switch the mode in. Defaults to the tf.get_default_session() value.
        """
        if sess is None:
            sess = tf.get_default_session()
        init_op = self._get_init_op(mode)
        sess.run(init_op, feed_dict)
