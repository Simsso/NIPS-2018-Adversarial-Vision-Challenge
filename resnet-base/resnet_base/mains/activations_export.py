from os import path

import scipy.io
import tensorflow as tf
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.baseline_resnet import BaselineResNet
from resnet_base.util.logger.factory import LoggerFactory
from resnet_base.util.logger.tf_logger_init import init as logger_init
from typing import Dict, List


tf.flags.DEFINE_string('activations_export_file', path.expanduser('~/.data/activations/baseline/act8_global_avg'),
                       'File to export the activations to, without file extension.')
FLAGS = tf.flags.FLAGS


def main(args) -> None:
    """
    This script feeds Tiny ImageNet samples into a ResNet and exports a file containing a dictionary. Each entry in the
    dictionary is a list of activations (or labels). The activations of a sample are only stored/exported, if the sample
    was classified correctly.
    """
    tf.reset_default_graph()
    tf.set_random_seed(15092017)
    logger_init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)
    with sess:
        # dataset
        pipeline = TinyImageNetPipeline(physical_batch_size=1, shuffle=False)
        imgs, labels = pipeline.get_iterator().get_next()

        # model
        logger_factory = LoggerFactory(num_valid_steps=1)
        model = BaselineResNet(logger_factory, imgs, labels)

        # init and restore pre-trained weights
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init, feed_dict=model.init_feed_dict)
        model.restore(sess)

        gather_activations(sess, pipeline, model, tf.estimator.ModeKeys.TRAIN)


def gather_activations(sess: tf.Session, pipeline: TinyImageNetPipeline, model: BaselineResNet,
                       mode: tf.estimator.ModeKeys, only_correct_ones: bool = True) -> None:
    """
    Feeds samples of the given mode through the given model and accumulates the activation values for correctly
    classified samples. Writes the activations to .mat files.
    """
    values_per_file = 100000
    n = pipeline.get_num_samples(mode)

    pipeline.switch_to(mode)
    export_tensors = model.activations.copy()
    export_tensors['target_labels'] = model.labels

    def get_blank_export_vals() -> Dict:
        blank_dict = {}
        for k in export_tensors.keys():
            blank_dict[k] = []
        return blank_dict

    export_vals = get_blank_export_vals()

    skipped_ctr, file_ctr = 0, 0
    for i in range(n):
        sample_export_val, accuracy = sess.run([export_tensors, model.accuracy])
        if accuracy < 1 and only_correct_ones:
            skipped_ctr += 1
            tf.logging.info("Skipping misclassified sample #{}".format(skipped_ctr))
        else:
            for key in sample_export_val.keys():
                export_vals[key].append(sample_export_val[key][0])  # unpack batches and push into storage
        tf.logging.info("Progress: {}/{}".format(i, n))
        if (i > 0 and i % values_per_file == 0) or (i+1) == n:
            save_activations(file_ctr, FLAGS.activations_export_file, export_vals)
            export_vals = get_blank_export_vals()
            file_ctr += 1


def save_activations(i: int, export_path: str, val_dict: Dict[str, any]) -> None:
    export_path = "{}_{}.mat".format(export_path, str(i).zfill(3))
    tf.logging.info("Exporting to {}".format(export_path))
    scipy.io.savemat(export_path, mdict=val_dict)


if __name__ == "__main__":
    tf.app.run()
