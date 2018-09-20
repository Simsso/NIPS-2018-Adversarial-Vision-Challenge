import tensorflow as tf
import numpy as np

from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.resnet import ResNet
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline

FLAGS = tf.flags.FLAGS


class ResNetTrainer(BaseTrainer):
    def __init__(self, sess: tf.Session, model: ResNet, pipeline: TinyImageNetPipeline):
        super().__init__(sess, model)
        self.resnet = model
        self.pipeline = pipeline
        self.__build_train_op()

    def __build_train_op(self):
        """
        Creates a train_op property that performs one step of minimizing the loss of the model.
        In this case, the AdamOptimizer with the specified FLAGS.learning_rate is used.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.train_op = optimizer.minimize(self.resnet.loss)

    def train_epoch(self):
        batch_size = FLAGS.train_batch_size
        num_samples = TinyImageNetPipeline.num_train_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.TRAIN)
        accuracy_mean, loss_mean = ResNetTrainer.__generic_epoch_with_params(batch_size, num_samples,
                                                                             batch_step=self.train_step)
        tf.logging.info("Training metrics: accuracy = {}, loss = {}".format(accuracy_mean, loss_mean))
        self.sess.run(self.model.increment_current_epoch)

    def train_step(self) -> (float, float):
        """
        Performs one training step (i.e. one batch) - and returns the batch's accuracy and loss.
        :return: a tuple of (accuracy, loss) for this training step
        """
        _, accuracy, loss = self.sess.run([self.train_op, self.resnet.accuracy, self.resnet.loss],
                                          feed_dict={
                                              self.resnet.is_training: True
                                          })
        return accuracy, loss

    def val_epoch(self):
        batch_size = FLAGS.val_batch_size
        num_samples = TinyImageNetPipeline.num_valid_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        accuracy_mean, loss_mean = ResNetTrainer.__generic_epoch_with_params(batch_size, num_samples,
                                                                             batch_step=self.val_step)
        tf.logging.info("Validation metrics: accuracy = {}, loss = {}".format(accuracy_mean, loss_mean))

    def val_step(self) -> (float, float):
        """
        Performs one validation step (i.e. one batch) - and returns the batch's accuracy and loss.
        :return: a tuple of (accuracy, loss) for this validation step
        """
        accuracy, loss = self.sess.run([self.resnet.accuracy, self.resnet.loss],
                                       feed_dict={
                                           self.resnet.is_training: False
                                       })
        return accuracy, loss

    @staticmethod
    def __generic_epoch_with_params(batch_size: int, num_samples: int, batch_step) -> (float, float):
        """
        Runs one epoch with the given parameters. Calls the given step function for each batch.
        :param batch_size: the number of samples used at every step
        :param num_samples: the total size of the data set
        :param batch_step: a function that runs and returns the accuracy and loss values for the batch
        :return: a tuple of (accuracy, loss), averaged over all batches
        """
        metrics = []
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            accuracy, loss = batch_step()
            metrics.append((accuracy, loss))

        accuracy_mean, loss_mean = np.mean(metrics, axis=0)
        return accuracy_mean, loss_mean
