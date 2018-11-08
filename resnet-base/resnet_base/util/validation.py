from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.baseline_lesci_resnet import BaselineLESCIResNet

LESCIMetrics = namedtuple("LESCIMetrics", ['accuracy', 'loss', 'accuracy_projection', 'accuracy_identity',
                                           'percentage_identity_mapped'])


def run_validation(model: BaselineLESCIResNet, pipeline: TinyImageNetPipeline, mode: tf.estimator.ModeKeys,
                   verbose: bool = False, batch_size: int = 100) -> LESCIMetrics:
    """
    Feeds all validation/train samples through the model and report classification accuracy and loss.
    :return: a LESCIMetrics tuple of the following (float-) values:
            - accuracy: mean overall accuracy
            - loss: mean overall loss
            - accuracy_projection: accuracy mean of the projected samples
            - accuracy_identity: accuracy mean of the identity-mapped samples
            - percentage_identity_mapped: percentage of inputs that have been identity-mapped
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

        if verbose:
            tf.logging.info("Starting evaluation")
        vals = []
        acc_mean_val, loss_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val = 0., 0., 0., 0., 0.
        num_samples = pipeline.get_num_samples(mode)
        n = num_samples // batch_size
        missed_samples = num_samples % batch_size
        if missed_samples > 0 and verbose:
            tf.logging.warning("Omitting {} samples because the batch size ({}) is not a divisor of the number of "
                               "samples ({}).".format(missed_samples, num_samples, batch_size))

        fetches = [model.accuracy, model.loss, model.accuracy_projection, model.accuracy_identity,
                   model.percentage_identity_mapped]
        for i in range(n):
            vals.append(sess.run(fetches))
            acc_mean_val, loss_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val = np.mean(vals, axis=0)
            if verbose:
                tf.logging.info("[{:,}/{:,}]\tCurrent overall accuracy: {:.3f}\tprojection: {:.3f}\tid-mapping: {:.3f}"
                                "\tpercentage id-mapped: {:.3f}"
                                .format(i, n, acc_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val))

        if verbose:
            tf.logging.info("[Done] Mean: accuracy {:.3f}, projection accuracy {:.3f}, identity mapping accuracy "
                            "{:.3f}, loss {:.3f}, id-mapped {:.3f}"
                            .format(acc_mean_val, acc_proj_mean_val, acc_id_mean_val, loss_mean_val,
                                    id_mapped_mean_val))
    return LESCIMetrics(acc_mean_val, loss_mean_val, acc_proj_mean_val, acc_id_mean_val, id_mapped_mean_val)
