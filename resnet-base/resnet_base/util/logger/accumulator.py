import numpy as np
import tensorflow as tf


class Accumulator:
    """
    Abstract class for TensorBoard logging accumulators. An accumulator aggregates multiple values of the same kind and
    reduces them into a `tf.Summary.Value` object which can then be written to TensorBoard.
    """

    def __init__(self, name: str, log_frequency: int):
        """
        :param name: Name of the accumulator which will appear in TensorBoard
        :param log_frequency: Lowest number of values to aggregate until writing to TensorBoard
        """
        self._values = []
        self._name = name
        self._log_frequency = log_frequency
        self._non_increments_added = 0

    def log_ready(self):
        """
        :return: `True` if enough values have been accumulated to write, `False` otherwise.
        """
        return len(self._values) - self._non_increments_added >= self._log_frequency

    def add(self, val: any, increment: bool = True) -> None:
        """
        Adds a new value to the accumulator. Must be called with values of the same type.
        :param val: Value to add
        :param increment: Whether or not to increment the step counter.
        """
        if not increment:
            self._non_increments_added += 1
        self._values.append(val)

    def _reduce(self) -> any:
        """
        Reduces the accumulated values into a single value. Implementation depends on the type of values.
        :return: Accumulated value
        """
        raise NotImplementedError()

    def to_summary_value(self) -> tf.Summary.Value:
        """
        Converts the accumulator into a summary value and deletes the accumulated values in order to collect a new list
        of values.
        :return: Summary value which can be added to a `tf.Summary` object
        """
        summary_value = self._get_summary_value()
        self.__flush()
        return summary_value

    def _get_summary_value(self) -> tf.Summary.Value:
        """
        :return: Summary value which can be added to a `tf.Summary` object
        """
        raise NotImplementedError()

    def __flush(self) -> None:
        """
        Removes all accumulated values.
        """
        self._values = []
        self._non_increments_added = 0


class ScalarAccumulator(Accumulator):
    """
    Accumulator for scalar values, e.g. loss or accuracy.
    """

    def _reduce(self) -> any:
        """
        Reduces the accumulated values into a single value by computing the mean.
        :return: Accumulated value
        """
        return np.mean(self._values)

    def _get_summary_value(self) -> tf.Summary.Value:
        """
        :return: Scalar summary value which can be added to a `tf.Summary` object
        """
        average = self._reduce()
        tf.logging.debug("{}: {}".format(self._name, average))
        return tf.Summary.Value(tag=self._name, simple_value=average)


class ScalarSumAccumulator(ScalarAccumulator):
    """
    Accumulator for scalar values which are a sum depending on the batch size.
    """

    def _reduce(self) -> any:
        """
        Reduces the accumulated values into a single value by computing the mean corrected by the number of increments.
        :return: Accumulated value
        """
        return np.sum(self._values) / float(len(self._values) - self._non_increments_added)


class HistogramAccumulator(Accumulator):
    """
    Accumulator for histogram values, e.g. embedding space usage.
    """

    def _reduce(self) -> any:
        """
        Reduces the accumulated values into a single value by computing the mean.
        :return: Accumulated value
        """
        return np.mean(np.array(self._values), axis=0)

    def _get_summary_value(self) -> tf.Summary.Value:
        """
        :return: Histogram summary value which can be added to a `tf.Summary` object
        """
        values = self._reduce()
        tf.logging.debug("{}: {}".format(self._name, values))
        counts, bin_edges = np.histogram(values, bins=1000)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        return tf.Summary.Value(tag=self._name, histo=hist)


class HistogramSumAccumulator(HistogramAccumulator):
    def _reduce(self) -> any:
        """
        Reduces the accumulated values into a single value by computing the mean corrected by the number of increments.
        :return: Accumulated value
        """
        return np.sum(np.array(self._values), axis=0) / float(len(self._values) - self._non_increments_added)
