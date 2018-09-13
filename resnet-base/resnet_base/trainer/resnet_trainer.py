import tensorflow as tf
from resnet_base.trainer.base_trainer import BaseTrainer
from resnet_base.model.resnet import ResNet


class ResNetTrainer(BaseTrainer):
    def __init__(self, sess: tf.Session, model: ResNet):
        super().__init__(sess, model)

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def val_epoch(self):
        raise NotImplementedError

    def val_step(self):
        raise NotImplementedError
