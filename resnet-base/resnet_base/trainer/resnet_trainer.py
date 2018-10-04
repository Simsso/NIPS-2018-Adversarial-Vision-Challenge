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
        accum_vars = [tf.get_variable('{}/accum_vars'.format(var.op.name), var.shape, tf.float32, tf.zeros_initializer,
                                      trainable=False) for var in var_list]
        self.zero_gradients_op = [var.assign(tf.zeros_like(var)) for var in accum_vars]
        gradients = optimizer.compute_gradients(loss=self.resnet.loss, var_list=var_list,
                                                aggregation_method=tf.AggregationMethod.ADD_N)

        # insert UPDATE_OPS if needed
        self.accumulate_gradients_op = [accum_vars[i].assign_add(g[0]) for i, g in enumerate(gradients)]

        grad_scaling = 1. / self.virtual_batch_size_factor
        self.apply_gradients_op = optimizer.apply_gradients([
            (tf.multiply(accum_vars[i], grad_scaling),  # accumulated, averaged gradients
             g[1])  # variable to update
            for i, g in enumerate(gradients)])

    def train_epoch(self) -> None:
        self.pipeline.switch_to(tf.estimator.ModeKeys.TRAIN)
        num_samples = TinyImageNetPipeline.num_train_samples
        num_virtual_batches = num_samples // (self.pipeline.batch_size * self.virtual_batch_size_factor)
        for _ in range(num_virtual_batches):
            self.train_step()
        self.sess.run(self.model.increment_current_epoch)

    def train_step(self):
        """
        Performs one training step (i.e. one virtual batch). One batch may consist of several physical batches making up
        one virtual batch. The number of real batches per virtual batch is specified by the virtual_batch_size_factor.
        """
        self.sess.run(self.zero_gradients_op)
        for i in range(self.virtual_batch_size_factor):
            vals = self.sess.run([self.accumulate_gradients_op] + self.train_logger.tensors,
                                 feed_dict={self.resnet.is_training: True})[1:]
            self.train_logger.step_completed(vals, increment=(i == 0))  # increment only once per virtual batch
        self.sess.run([self.apply_gradients_op, self.model.increment_global_step])  # update model weights

    def val_epoch(self) -> None:
        self.pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        num_physical_batches = TinyImageNetPipeline.num_valid_samples // self.pipeline.batch_size
        for _ in range(num_physical_batches):
            self.val_step()

    def val_step(self) -> None:
        """
        Performs one validation step (i.e. one physical batch).
        """
        vals = self.sess.run(self.valid_logger.tensors, feed_dict={self.resnet.is_training: False})
        self.valid_logger.step_completed(vals)
