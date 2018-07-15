import tensorflow as tf
import data.tiny_imagenet as tiny_imagenet


def main(args=None):
    tiny_imagenet.batch_q('train', {'num_epochs': 1, 'batch_size': 100})


if __name__ == '__main__':
    tf.app.run()
