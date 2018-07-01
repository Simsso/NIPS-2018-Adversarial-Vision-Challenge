import attack as attack
import tensorflow as tf
import train
import linear_combination as lc
import layerwise_perturbation as pert


def main(args=None):
    """
    This module package contains functionality for three different things:
    * training a CNN on MNISt and storing the weights (+ logging to TensorBoard)
    * loading the weights and analyzing the classification of linear combinations between inputs
    * loading the weights and computing an adversarial example using FGSM
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    # train.train()
    # lc.run_analysis(train.MODEL_NAME)
    # attack.fgsm(train.MODEL_NAME)
    img, adv = attack.get_attack_batch(train.MODEL_NAME, 10)
    pert.run_analysis(train.MODEL_NAME, img, adv)


if __name__ == '__main__':
    tf.app.run()
