from resnet_base.util.logger.accumulator import Accumulator
import tensorflow as tf
from typing import List

FLAGS = tf.flags.FLAGS


class Logger:
    """
    A logger instance can be used to log to TensorBoard. Opposed to common `tf.summary.` calls, it uses accumulators
    to create summaries over multiple batches.
    """
    def __init__(self, sess: tf.Session, log_dir: str, global_step: tf.Tensor):
        """
        :param sess: TensorFlow session
        :param log_dir: Directory to log the TensorBoard log output to
        """
        self.__log_dir = log_dir
        self.__writer = tf.summary.FileWriter(log_dir, sess.graph)
        self.__sess = sess
        self.tensors = []
        self.__accumulators = []
        self.__global_step = global_step

    def add(self, tensor: tf.Tensor, acc: Accumulator) -> None:
        """
        Adds a new tensor to the logger. The accumulator defines how the tensor will be logged to TensorBoard, e.g. a
        scalar or a histogram.
        :param tensor: Tensor to log
        :param acc: Accumulator to use
        """
        self.tensors.append(tensor)
        self.__accumulators.append(acc)

    def __update_accumulators(self, vals: List[any]) -> None:
        """
        Updates the accumulators by adding a new value to each of them.
        :param vals: New values to add (each entry in the list belongs to one entry in the `self.tensors` list.
        """
        assert len(vals) == len(self.tensors)
        for i, val in enumerate(vals):
            self.__accumulators[i].add(val)

    def step_completed(self, vals: List[any]) -> None:
        """
        Called after completion of a step / batch forward pass.
        :param vals: One value for each entry in the self.tensors list. Commonly called with the return value of
                     `sess.run(logger.tensors)`.
        """
        self.__update_accumulators(vals)
        summary_values = []
        for acc in self.__accumulators:
            # check whether the accumulator contains enough values to be written out to the log
            if acc.log_ready():
                summary_values.append(acc.to_summary_value())
        if len(summary_values) > 0:
            tf.logging.info("Writing custom summary object to '{}'".format(self.__log_dir))
            summary = tf.Summary(value=summary_values)
            self.__writer.add_summary(summary, global_step=self.__global_step.eval(session=self.__sess))
