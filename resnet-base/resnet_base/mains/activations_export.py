from os import path

import scipy.io

import tensorflow as tf

from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.activations_resnet import ActivationsResNet
from resnet_base.util.logger.factory import LoggerFactory
from resnet_base.util.logger.tf_logger_init import init as logger_init
from typing import Dict


tf.flags.DEFINE_string('activations_export_file', path.expanduser('~/.data/activations/data_100k_act5_block3.mat'),
                       'File to export the activations to.')
FLAGS = tf.flags.FLAGS


def main(args):
    """
    This script feeds Tiny ImageNet samples into a ResNet and exports a file containing a dictionary. Each entry in the
    dictionary is a list of activations (or labels).
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
        model = ActivationsResNet(logger_factory, imgs, labels)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # use training data
        train = tf.estimator.ModeKeys.TRAIN
        pipeline.switch_to(train)
        gather_activations(sess, pipeline, model, train)


def gather_activations(sess: tf.Session, pipeline: TinyImageNetPipeline, model: ActivationsResNet,
                       mode: tf.estimator.ModeKeys) -> None:
    pipeline.switch_to(mode)
    n = pipeline.get_num_samples(mode)
    export_tensors = model.activations.copy()
    export_tensors['out_labels'] = model.labels

    export_vals = {}
    for key in export_tensors.keys():
        export_vals[key] = []

    for i in range(n):
        sample_export_val = sess.run(export_tensors)
        for key in sample_export_val.keys():
            export_vals[key].append(sample_export_val[key][0])  # unpack batches and push into storage
        print("Progress: {}/{}".format(i, n))
    save_activations(FLAGS.activations_export_file, export_vals)


def save_activations(export_path: str, val_dict: Dict[str, any]) -> None:
    scipy.io.savemat(export_path, mdict=val_dict)


if __name__ == "__main__":
    tf.app.run()
