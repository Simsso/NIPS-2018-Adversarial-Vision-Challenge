from nips_defense.util.logger.accumulator import Accumulator, ScalarAccumulator, HistogramAccumulator, \
    HistogramSumAccumulator
from nips_defense.util.logger.logger import Logger
import tensorflow as tf


class LoggerFactory:
    def __init__(self, num_valid_steps: int = 1):
        self.__log_elements = []
        self.__num_valid_steps = num_valid_steps

    def add_scalar(self, name: str, tensor: tf.Tensor, log_frequency: int = 1) -> None:
        """
        Adds a scalar tensor to the factory.
        :param name: Display name of the tensor
        :param tensor: The tensor
        :param log_frequency: Lowest number of values to aggregate until writing to TensorBoard

        """
        self.__log_elements.append(('scalar', name, tensor, log_frequency))

    def add_histogram(self, name: str, tensor: tf.Tensor, is_sum_value: bool = False, log_frequency: int = 1) -> None:
        """
        Adds a histogram tensor to the factory.
        :param name: Display name of the tensor
        :param tensor: The tensor
        :param is_sum_value: True if the value is a sum depending on the batch size
        :param log_frequency: Lowest number of values to aggregate until writing to TensorBoard
        """
        log_type = 'histogram_sum' if is_sum_value else 'histogram'
        self.__log_elements.append((log_type, name, tensor, log_frequency))

    def __create_accumulators(self, log_type: str, name: str, log_frequency: int) -> (Accumulator, Accumulator):
        if log_type == 'scalar':
            constructor = ScalarAccumulator
        elif log_type == 'histogram':
            constructor = HistogramAccumulator
        elif log_type == 'histogram_sum':
            constructor = HistogramSumAccumulator
        else:
            raise ValueError("Parameter 'log_type' is invalid. Got '{}'.".format(log_type))
        return constructor(name, log_frequency), constructor(name, self.__num_valid_steps)

    def create_loggers(self, sess: tf.Session, train_log_dir: str, valid_log_dir: str, global_step: tf.Tensor) \
            -> (Logger, Logger):
        train_logger = Logger(sess, train_log_dir, global_step)
        valid_logger = Logger(sess, valid_log_dir, global_step)
        for log_type, name, tensor, log_freq in self.__log_elements:
            train_accumulator, valid_accumulator = self.__create_accumulators(log_type, name, log_freq)
            train_logger.add(tensor, train_accumulator)
            valid_logger.add(tensor, valid_accumulator)
        return train_logger, valid_logger
