import attack as attack
import tensorflow as tf
import train
import linear_combination as lc


def main(args=None):
    """
    This module package contains functionality for three different things:
    * training a CNN on MNISt and storing the weights (+ logging to TensorBoard)
    * loading the weights and analyzing the classification of linear combinations between inputs
    * loading the weights and computing an adversarial example using FGSM
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    train.train()
    lc.run_analysis(train.MODEL_NAME)
    attack.fgsm(train.MODEL_NAME)


if __name__ == '__main__':
    tf.app.run()
