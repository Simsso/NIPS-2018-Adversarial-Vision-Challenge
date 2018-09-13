import tensorflow as tf
from resnet_base.model.resnet import ResNet
from resnet_base.trainer.resnet_trainer import ResNetTrainer


def main(args):
    model = ResNet()
    sess = tf.Session()

    trainer = ResNetTrainer(sess, model)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
