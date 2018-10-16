from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
import numpy as np
import tensorflow as tf
from resnet_base.model.resnet import ResNet
from resnet_base.model.pca_resnet import PCAResNet

VALIDATION_BATCH_SIZE = 10  # adjustment based on available RAM
tf.logging.set_verbosity(tf.logging.DEBUG)


def run_validation(model: ResNet, pipeline: TinyImageNetPipeline):
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    with sess.as_default():
        sess.run(init, feed_dict=model.init_feed_dict)
        model.restore(sess)

        pipeline.switch_to(tf.estimator.ModeKeys.EVAL)

        tf.logging.info("Starting evaluation")
        vals = []
        acc_mean_val, loss_mean_val = 0, 0
        n = TinyImageNetPipeline.num_valid_samples // VALIDATION_BATCH_SIZE
        for i in range(n):
            vals.append(sess.run([model.accuracy, model.loss]))
            acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
            tf.logging.info("[{}/{}]Current accuracy: {}".format(i, n, acc_mean_val))
        tf.logging.info("Final validation data: accuracy {}, loss {}".format(acc_mean_val, loss_mean_val))


def main(args):
    pipeline = TinyImageNetPipeline(physical_batch_size=VALIDATION_BATCH_SIZE)
    imgs, labels = pipeline.get_iterator().get_next()
    model = PCAResNet(x=imgs, labels=labels)
    run_validation(model, pipeline)


if __name__ == "__main__":
    tf.app.run()
