import tensorflow as tf
from resnet_base.model.resnet import ResNet
from resnet_base.trainer.resnet_trainer import ResNetTrainer


def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    model = ResNet()
    trainer = ResNetTrainer(sess, model)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
