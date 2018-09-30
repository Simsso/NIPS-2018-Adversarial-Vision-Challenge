import tensorflow as tf
from collections import namedtuple

from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.vq_resnet import VQResNet
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.util.logger.accumulator import ScalarAccumulator, HistogramAccumulator

Metrics = namedtuple('Metrics', 'accuracy loss')
FLAGS = tf.flags.FLAGS


class ResNetTrainer(BaseTrainer):
    def __init__(self, model: VQResNet, pipeline: TinyImageNetPipeline):
        super().__init__(model)
        self.resnet = model
        self.pipeline = pipeline
        self.__build_train_op()
        self.__init_loggers()

    def __init_loggers(self) -> None:
        train_acc = 10
        self.__register_log_tensor(self.resnet.loss, ScalarAccumulator, 'loss', train_acc)
        self.__register_log_tensor(self.resnet.accuracy, ScalarAccumulator, 'accuracy', train_acc)
        self.__register_log_tensor(self.resnet.vq_access_count, HistogramAccumulator, 'access_count', train_acc)

    def __register_log_tensor(self, tensor: tf.Tensor, accumulator_class, name: str, train_accumulation: int) -> None:
        self.train_logger.add(tensor, accumulator_class(name, train_accumulation))
        valid_steps = TinyImageNetPipeline.num_valid_samples // self.pipeline.batch_size
        self.validation_logger.add(tensor, accumulator_class(name, valid_steps))

    def __build_train_op(self) -> None:
        """
        Creates a train_op property that performs one step of minimizing the loss of the model.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.resnet.custom_scope.name)
            self.train_op = optimizer.minimize(self.resnet.loss, var_list=var_list)

    def train_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_train_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.TRAIN)
        ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples, batch_step=self.train_step)

    def train_step(self):
        """
        Performs one training step (i.e. one batch).
        """
        vals = self.sess.run([self.train_op] + self.train_logger.tensors, feed_dict={self.resnet.is_training: True})[1:]
        self.train_logger.step_completed(vals)

    def val_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_valid_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples, batch_step=self.val_step)

    def val_step(self) -> None:
        """
        Performs one validation step (i.e. one batch).
        """
        vals = self.sess.run(self.validation_logger.tensors, feed_dict={self.resnet.is_training: False})
        self.validation_logger.step_completed(vals)

    @staticmethod
    def __generic_epoch_with_params(batch_size: int, num_samples: int, batch_step):
        """
        Runs one epoch with the given parameters. Calls the given step function for each batch.
        :param batch_size: the number of samples used at every step
        :param num_samples: the total size of the data set
        :param batch_step: a function that runs and returns the accuracy and loss metrics for the batch
        """
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            batch_step()
