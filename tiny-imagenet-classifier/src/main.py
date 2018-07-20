import tensorflow as tf
import trainer.sgd as sgd_trainer
import trainer.sgd_resnet as resnet_trainer


def main(args=None):
    resnet_trainer.train()


if __name__ == '__main__':
    tf.app.run()
