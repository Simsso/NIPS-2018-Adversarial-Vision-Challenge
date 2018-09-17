from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
import numpy as np
import os
import tensorflow as tf
from resnet_base.model.resnet import ResNet

VALIDATION_BATCH_SIZE = 1024  # does not affect training results; adjustment based on GPU RAM
TF_LOGS = os.path.join('..', 'tf_logs')

tf.logging.set_verbosity(tf.logging.DEBUG)


def run_validation(model: ResNet):
    pipeline = TinyImageNetPipeline()

    # data set
    valid_batch = pipeline.get_iterator().get_next()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    with sess.as_default():
        sess.run(init)
        pipeline.switch_to(tf.estimator.ModeKeys.EVAL)
        model.load(sess)

        tf.logging.info("Starting evaluation")
        vals = []
        acc_mean_val, loss_mean_val = 0, 0
        for _ in range(min(TinyImageNetPipeline.num_valid_samples // VALIDATION_BATCH_SIZE, TinyImageNetPipeline.num_valid_samples)):
            valid_images, valid_labels = sess.run(valid_batch)
            vals.append(sess.run([model.accuracy, model.loss],
                                 feed_dict={model.x: valid_images, model.labels: valid_labels}))
            acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
            tf.logging.info("Current accuracy: {}".format(acc_mean_val))
        tf.logging.info("Final validation data: accuracy {}, loss {}".format(acc_mean_val, loss_mean_val))


def main(args):
    model = ResNet()
    run_validation(model)


if __name__ == "__main__":
    tf.app.run()
