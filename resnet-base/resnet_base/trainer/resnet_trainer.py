import tensorflow as tf
from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.resnet import ResNet
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline

FLAGS = tf.flags.FLAGS


class ResNetTrainer(BaseTrainer):
    def __init__(self, model: ResNet, pipeline: TinyImageNetPipeline, virtual_batch_size_factor: int = 1):
        super().__init__(model)
        self.virtual_batch_size_factor = virtual_batch_size_factor
        self.resnet = model
        self.pipeline = pipeline
        self.__build_train_op()

    def __build_train_op(self) -> None:
        """
        Creates three ops, namely accumulate_gradients_op, apply_gradients_op, and zero_gradients_op.
        accumulate_gradients_op should be evaluated several times (virtual batch size factor) before applying the
        accumulated gradients to the network weights, i.e. weight update.
        zero_gradients_op is used to reset the gradient accumulator after every weight update.
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        # train only custom variables
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.resnet.custom_scope.name)
        accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in var_list]
        self.zero_gradients_op = [var.assign(tf.zeros_like(var)) for var in accum_vars]
        gradients = optimizer.compute_gradients(self.resnet.loss, var_list)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.accumulate_gradients_op = [accum_vars[i].assign_add(g[0]) for i, g in enumerate(gradients)]

        self.apply_gradients_op = optimizer.apply_gradients([(accum_vars[i], g[1]) for i, g in enumerate(gradients)])

    def train_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_train_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.TRAIN)
        ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples, batch_step=self.train_step)

    def train_step(self):
        """
        Performs one training step (i.e. one batch). One batch may consist of several batches making up one virtual
        batch. The number of real batches per virtual batch is specified by the virtual_batch_size_factor.
        """
        self.sess.run(self.zero_gradients_op)
        for i in range(self.virtual_batch_size_factor):
            vals = self.sess.run([self.accumulate_gradients_op] + self.train_logger.tensors,
                                 feed_dict={self.resnet.is_training: True})[1:]
            self.train_logger.step_completed(vals, increment=(i == 0))  # increment only once per virtual batch
        self.sess.run(self.apply_gradients_op)  # update model weights

    def val_epoch(self) -> None:
        num_samples = TinyImageNetPipeline.num_valid_samples

        self.pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        ResNetTrainer.__generic_epoch_with_params(self.pipeline.batch_size, num_samples, batch_step=self.val_step)

    def val_step(self) -> None:
        """
        Performs one validation step (i.e. one batch).
        """
        vals = self.sess.run(self.valid_logger.tensors, feed_dict={self.resnet.is_training: False})
        self.valid_logger.step_completed(vals)

    @staticmethod
    def __generic_epoch_with_params(batch_size: int, num_samples: int, batch_step):
        """
        Runs one epoch with the given parameters. Calls the given step function for each batch.
        :param batch_size: the number of samples used at every step
        :param num_samples: the total size of the data set
        :param batch_step: a function that runs the batch
        """
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            batch_step()
