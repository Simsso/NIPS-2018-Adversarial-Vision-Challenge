import tensorflow as tf
import trainer.sgd as sgd_trainer
import data.tiny_imagenet as data


def main(args=None):
    sgd_trainer.train()


if __name__ == '__main__':
    tf.app.run()
