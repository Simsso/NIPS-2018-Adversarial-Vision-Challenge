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
