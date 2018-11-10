import tensorflow as tf

from nips_defense.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from nips_defense.model.baseline_lesci_resnet import BaselineLESCIResNet
from nips_defense.util.validation import run_validation

BATCH_SIZE = 100  # adjustment based on available RAM
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    pipeline = TinyImageNetPipeline(physical_batch_size=BATCH_SIZE, shuffle=False)
    imgs, labels = pipeline.get_iterator().get_next()
    model = BaselineLESCIResNet(x=imgs, labels=labels)
    run_validation(model, pipeline, mode=tf.estimator.ModeKeys.EVAL)


if __name__ == "__main__":
    tf.app.run()
