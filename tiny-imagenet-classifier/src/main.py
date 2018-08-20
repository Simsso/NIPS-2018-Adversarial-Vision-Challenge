import tensorflow as tf
import trainer.adam_transfer as trainer
import model.inception_transfer as model


def main(args=None):
    trainer.train(model)


if __name__ == '__main__':
    tf.app.run()
