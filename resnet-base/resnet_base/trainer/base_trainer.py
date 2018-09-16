import tensorflow as tf
from resnet_base.model.base_model import BaseModel


class BaseTrain:
    def __init__(self, sess: tf.Session, model: BaseModel, config):
        # assign all class attributes
        self.model = model
        self.config = config
        self.sess = sess

    def init_variables(self):
        # initialize all variables of the graph
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):
        start_epoch = self.model.current_epoch.eval(self.sess)
        for cur_epoch in range(start_epoch, self.config.num_epochs + 1):
            self.train_epoch()
            self.sess.run(self.model.increment_current_epoch)

    def train_epoch(self, epoch=None):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary

        :param epoch: take the number of epoch if you are interested
        :return:
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step

        - run the tensorflow session
        :return: any metrics you need to summarize
        """
        raise NotImplementedError
