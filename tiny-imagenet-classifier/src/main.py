import tensorflow as tf
import trainer.sgd as trainer
#import trainer.adam_resnet as trainer


def main(args=None):
    trainer.train()


if __name__ == '__main__':
    tf.app.run()
