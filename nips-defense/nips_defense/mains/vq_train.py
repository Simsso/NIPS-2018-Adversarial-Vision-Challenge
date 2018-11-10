import tensorflow as tf
from nips_defense.model.parallel_vq_resnet import ParallelVQResNet
from nips_defense.trainer.resnet_trainer import ResNetTrainer
from nips_defense.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from nips_defense.util.logger.factory import LoggerFactory
from nips_defense.util.logger.tf_logger_init import init as logger_init


tf.flags.DEFINE_integer("physical_batch_size", 32, "Number of samples per batch that is fed through the GPU at once.")
tf.flags.DEFINE_integer("virtual_batch_size_factor", 8, "Number of batches per weight update.")
FLAGS = tf.flags.FLAGS


def main(args):
    tf.reset_default_graph()
    tf.set_random_seed(15092017)
    logger_init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)
    with sess:
        # dataset
        pipeline = TinyImageNetPipeline(physical_batch_size=FLAGS.physical_batch_size)
        imgs, labels = pipeline.get_iterator().get_next()

        # model
        logger_factory = LoggerFactory(num_valid_steps=TinyImageNetPipeline.num_valid_samples // pipeline.batch_size)
        model = ParallelVQResNet(logger_factory, imgs, labels)

        # training
        trainer = ResNetTrainer(model, pipeline, FLAGS.virtual_batch_size_factor)
        trainer.train()


if __name__ == "__main__":
    tf.app.run()
