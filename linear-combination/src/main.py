import attack_fgsm as attack
import tensorflow as tf
import train
import linear_combination as lc


def main(args=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    #train.train()
    lc.run_analysis(train.MODEL_NAME)
    attack.run_fgsm(train.MODEL_NAME)


if __name__ == '__main__':
    tf.app.run()
