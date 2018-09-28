from resnet_base.util.logger.accumulator import Accumulator
import tensorflow as tf
from typing import List

FLAGS = tf.flags.FLAGS


class Logger:
    def __init__(self, sess: tf.Session, log_dir: str):
        self.step_count = 0
        self.writer = tf.summary.FileWriter(log_dir, sess.graph)
        self.sess = sess
        self.tensors = []
        self.accumulators = []

    def add(self, tensor: tf.Tensor, acc: Accumulator) -> None:
        self.tensors.append(tensor)
        self.accumulators.append(acc)

    def update_accumulators(self, vals: List[any]):
        for i, val in enumerate(vals):
            self.accumulators[i].add(val)

    def step_completed(self, vals: List[any]):
        self.step_count += 1
        self.update_accumulators(vals)


class TrainingLogger(Logger):
    def step_completed(self, vals: List[any]):
        super(TrainingLogger, self).step_completed(vals)
        summary_values = []
        for acc in self.accumulators:
            if len(acc.values) >= acc.log_frequency:
                summary_values.append(acc.to_summary_value())
                acc.flush()
        summary = tf.Summary(value=summary_values)
        self.writer.add_summary(summary, global_step=self.step_count)


class ValidationLogger(Logger):
    def step_completed(self, vals: List[any]):
        return super().step_completed(vals)
