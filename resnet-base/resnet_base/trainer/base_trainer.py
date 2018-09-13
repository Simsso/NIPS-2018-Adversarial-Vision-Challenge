import tensorflow as tf
from resnet_base.model.base_model import BaseModel

tf.flags.DEFINE_float("learning_rate", 1e-4, "The learning rate used for training.")
tf.flags.DEFINE_integer("num_epochs", 10, "The number of epochs for which training is performed.")
tf.flags.DEFINE_integer("train_batch_size", 64, "The batch size used when training.")
tf.flags.DEFINE_integer("val_batch_size", 512, "The batch size used when validating (optimize based on RAM).")

FLAGS = tf.flags.FLAGS


class BaseTrainer:
    def __init__(self, sess: tf.Session, model: BaseModel):
        self.model = model
        self.sess = sess

    def init_variables(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamic GPU memory allocation

        sess = tf.Session(config=config)
        with self.sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            sess.run(init)

            # restore weights (as specified in the FLAGS)
            self.model.load(sess)

            try:
                while not coord.should_stop():
                    # get the current epoch so we can re-start training from there
                    start_epoch = self.model.current_epoch.eval(self.sess)

                    for _ in range(start_epoch, FLAGS.num_epochs + 1):
                        self.train_epoch()
                        self.sess.run(self.model.increment_current_epoch)

                        # run validation epoch to monitor training
                        self.val_epoch()

                    break

            except tf.errors.OutOfRangeError as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

    def train_epoch(self):
        """
        Trains the model for one epoch.
        Should use the batch size defined in FLAGS.train_batch_size.
        """
        raise NotImplementedError

    def val_epoch(self):
        """
        Performs inference and calculates evaluation metrics for one full epoch of the validation set.
        Should use the batch size defined in FLAGS.val_batch_size.
        """
        raise NotImplementedError

