import numpy as np
import tensorflow as tf


class Accumulator:
    def __init__(self, name: str, log_frequency: int):
        self.values = []
        self.name = name
        self.log_frequency = log_frequency

    def add(self, val: any) -> None:
        self.values.append(val)

    def reduce(self) -> any:
        raise NotImplementedError()

    def to_summary_value(self) -> tf.Summary.Value:
        raise NotImplementedError()

    def flush(self) -> None:
        self.values = []


class ScalarAccumulator(Accumulator):
    def reduce(self) -> any:
        return np.mean(self.values)

    def to_summary_value(self) -> tf.Summary.Value:
        return tf.Summary.Value(tag=self.name, simple_value=self.reduce())


class HistogramAccumulator(Accumulator):
    def reduce(self) -> any:
        return np.mean(np.array(self.values), axis=0)

    def to_summary_value(self) -> tf.Summary.Value:
        values = self.reduce()
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

        return tf.Summary.Value(tag=self.name, histo=hist)
