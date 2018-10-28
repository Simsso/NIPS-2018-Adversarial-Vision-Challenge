from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.baseline_lesci_resnet import BaselineLESCIResNet

BATCH_SIZE = 100  # adjustment based on available RAM
tf.logging.set_verbosity(tf.logging.DEBUG)


def run_validation(model: BaselineLESCIResNet, pipeline: TinyImageNetPipeline, mode: tf.estimator.ModeKeys)\
        -> Tuple[float, float]:
    """
    Feeds all validation/train samples through the model and report classification accuracy and loss.
    :return: Dictionary of mean accuracy and mean loss
    """
    tf.logging.info("Running evaluation on with mode {}.".format(mode))
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    with sess.as_default():
        try:
            sess.run(init, feed_dict=model.init_feed_dict)
        except InvalidArgumentError:
            tf.logging.info("Could not execute the init op, trying to restore all variable.")
        model.restore(sess)

        pipeline.switch_to(mode)

        tf.logging.info("Starting evaluation")
        vals = []
        acc_mean_val, loss_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val = 0., 0., 0., 0., 0.
        num_samples = pipeline.get_num_samples(mode)
        n = num_samples // BATCH_SIZE
        missed_samples = num_samples % BATCH_SIZE
        if missed_samples > 0:
            tf.logging.warning("Omitting {} samples because the batch size ({}) is not a divisor of the number of "
                               "samples ({}).".format(missed_samples, num_samples, BATCH_SIZE))

        fetches = [model.accuracy, model.loss, model.accuracy_projection, model.accuracy_identity,
                   model.percentage_identity_mapped]
        for i in range(n):
            vals.append(sess.run(fetches))
            acc_mean_val, loss_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val = np.mean(vals, axis=0)
            tf.logging.info("[{:,}/{:,}]\tCurrent overall accuracy: {:.3f}\tprojection: {:.3f}\tid-mapping: {:.3f}"
                            "\tpercentage id-mapped: {:.3f}"
                            .format(i, n, acc_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val))

        tf.logging.info("[Done] Mean: accuracy {:.3f}, projection accuracy {:.3f}, identity mapping accuracy {:.3f}, "
                        "loss {:.3f}, id-mapped {:.3f}"
                        .format(acc_mean_val, acc_proj_mean_val, acc_id_mean_val, loss_mean_val, id_mapped_mean_val))
    return acc_mean_val, loss_mean_val


def main(args):
    pipeline = TinyImageNetPipeline(physical_batch_size=BATCH_SIZE, shuffle=False)
    imgs, labels = pipeline.get_iterator().get_next()
    model = BaselineLESCIResNet(x=imgs, labels=labels)
    run_validation(model, pipeline, mode=tf.estimator.ModeKeys.EVAL)


if __name__ == "__main__":
    tf.app.run()
