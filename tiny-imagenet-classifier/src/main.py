import tensorflow as tf
import trainer.sgd as sgd_trainer


def main(args=None):
    sgd_trainer.train()


if __name__ == '__main__':
    tf.app.run()
