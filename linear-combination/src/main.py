import tensorflow as tf
import train


def main(args=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    train.train()


if __name__ == '__main__':
    tf.app.run()
