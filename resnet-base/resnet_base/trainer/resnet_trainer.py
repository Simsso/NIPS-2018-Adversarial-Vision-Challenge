import tensorflow as tf
import numpy as np
from collections import namedtuple

from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.resnet import ResNet
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline

Metrics = namedtuple('Metrics', 'accuracy loss')
FLAGS = tf.flags.FLAGS


class ResNetTrainer(BaseTrainer):
    def __init__(self, sess: tf.Session, model: ResNet, pipeline: TinyImageNetPipeline):
        super().__init__(sess, model)
        self.resnet = model
        self.pipeline = pipeline
        self.__build_train_op()

    def __build_train_op(self) -> None:
        """
        Creates a train_op property that performs one step of minimizing the loss of the model.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.train_op = optimizer.minimize(self.resnet.loss)

    def train_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_train_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.TRAIN)
        metrics = ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples,
                                                            batch_step=self.train_step)
        tf.logging.info("Training metrics: accuracy = {}, loss = {}".format(metrics.accuracy, metrics.loss))

    def train_step(self) -> Metrics:
        """
        Performs one training step (i.e. one batch) - and returns the batch's accuracy and loss.
        :return: a Metrics tuple of (accuracy, loss) for this training step
        """
        _, accuracy, loss = self.sess.run([self.train_op, self.resnet.accuracy, self.resnet.loss],
                                          feed_dict={
                                              self.resnet.is_training: True
                                          })
        return Metrics(accuracy, loss)

    def val_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_valid_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        metrics = ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples,
                                                            batch_step=self.val_step)
        tf.logging.info("Validation metrics: accuracy = {}, loss = {}".format(metrics.accuracy, metrics.loss))

    def val_step(self) -> Metrics:
        """
        Performs one validation step (i.e. one batch) - and returns the batch's accuracy and loss.
        :return: a Metrics tuple of (accuracy, loss) for this validation step
        """
        accuracy, loss = self.sess.run([self.resnet.accuracy, self.resnet.loss],
                                       feed_dict={
                                           self.resnet.is_training: False
                                       })
        return Metrics(accuracy, loss)

    @staticmethod
    def __generic_epoch_with_params(batch_size: int, num_samples: int, batch_step) -> Metrics:
        """
        Runs one epoch with the given parameters. Calls the given step function for each batch.
        :param batch_size: the number of samples used at every step
        :param num_samples: the total size of the data set
        :param batch_step: a function that runs and returns the accuracy and loss metrics for the batch
        :return: a Metrics tuple of (accuracy, loss), averaged over all batches
        """
        metrics = []
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            batch_metrics = batch_step()
            metrics.append((batch_metrics.accuracy, batch_metrics.loss))

        accuracy_mean, loss_mean = np.mean(metrics, axis=0)
        return Metrics(accuracy_mean, loss_mean)
