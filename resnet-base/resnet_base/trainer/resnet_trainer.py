import tensorflow as tf
import numpy as np

from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.resnet import ResNet
import resnet_base.data.tiny_imagenet as data   # temporarily

FLAGS = tf.flags.FLAGS


class ResNetTrainer(BaseTrainer):
    def __init__(self, sess: tf.Session, model: ResNet):
        super().__init__(sess, model)
        self.resnet = model
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
        # TODO incorporate updated tf.data input pipeline when done, add summaries (abstracted in Logger class)
        batch_size = FLAGS.train_batch_size
        num_samples = data.NUM_TRAIN_SAMPLES
        batch_queue = data.batch_q('train', batch_size)

        accuracy_mean, loss_mean = ResNetTrainer.__generic_epoch_with_params(batch_size, num_samples, batch_queue,
                                                                             batch_step=self.train_step)
        tf.logging.info("Training metrics: accuracy = {}, loss = {}".format(accuracy_mean, loss_mean))
        self.sess.run(self.model.increment_current_epoch)

    def train_step(self, batch_queue) -> (float, float):
        """
        Performs one training step (i.e. one batch) - and returns the batch's accuracy and loss.
        :param batch_queue: the queue used to acquire the next batch (TODO other tf.data pipeline)
        :return: a tuple of (accuracy, loss) for this training step
        """
        images, labels = self.sess.run(batch_queue)
        _, accuracy, loss = self.sess.run([self.train_op, self.resnet.accuracy, self.resnet.loss],
                                          feed_dict={
                                              self.resnet.x: images,
                                              self.resnet.labels: labels,
                                              self.resnet.is_training: True
                                          })
        return accuracy, loss

    def val_epoch(self):
        batch_size = FLAGS.val_batch_size
        num_samples = data.NUM_VALIDATION_SAMPLES
        batch_queue = data.batch_q('val', batch_size)

        accuracy_mean, loss_mean = ResNetTrainer.__generic_epoch_with_params(batch_size, num_samples, batch_queue,
                                                                             batch_step=self.val_step)
        tf.logging.info("Validation metrics: accuracy = {}, loss = {}".format(accuracy_mean, loss_mean))

    def val_step(self, batch_queue) -> (float, float):
        """
        Performs one validation step (i.e. one batch) - and returns the batch's accuracy and loss.
        :param batch_queue: the queue used to acquire the next batch (TODO other tf.data pipeline)
        :return: a tuple of (accuracy, loss) for this validation step
        """
        images, labels = self.sess.run(batch_queue)
        accuracy, loss = self.sess.run([self.resnet.accuracy, self.resnet.loss],
                                       feed_dict={
                                           self.resnet.x: images,
                                           self.resnet.labels: labels,
                                           self.resnet.is_training: False
                                       })
        return accuracy, loss

    @staticmethod
    def __generic_epoch_with_params(batch_size: int, num_samples: int, batch_queue, batch_step)\
            -> (float, float):
        """
        Runs one epoch with the given parameters. Calls the given step function for each batch.
        :param batch_size: the number of samples used at every step
        :param num_samples: the total size of the data set
        :param batch_queue: the queue used to acquire the next batch (TODO other tf.data pipeline)
        :param batch_step: a function that takes a batch_queue and returns accuracy and loss values for the batch
        :return: a tuple of (accuracy, loss), averaged over all batches
        """
        metrics = []
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            accuracy, loss = batch_step(batch_queue)
            metrics.append((accuracy, loss))

        accuracy_mean, loss_mean = np.mean(metrics, axis=0)
        return accuracy_mean, loss_mean
